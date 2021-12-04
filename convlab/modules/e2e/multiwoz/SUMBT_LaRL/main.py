###
# End-to-end: SUMBT + LaRL
###
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import csv
import os
import logging
import argparse
import random
import collections
from collections import defaultdict
from tqdm import tqdm, trange
import json
import math

import pdb

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from tensorboardX import SummaryWriter

from utils.argument import add_parse_option
#from utils.processor import Processor, get_label_embedding, get_sent, get_bert_sent
from utils.processor_v3 import Processor, get_label_embedding, get_sent, get_bert_sent
from utils.utils import LossManager, save_configure, warmup_linear, get_key_list
from model import SUMBTLaRL
#from model_v2 import SUMBTLaRL

from utils.task_generate import task_generate, generate
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.evaluators import MultiWozEvaluator

from latent_dialog.rl_agents import OfflineLatentRlAgent
from convlab.modules.dst.multiwoz.dst_util import init_state
import time
    
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # parse options
    parser = argparse.ArgumentParser()
    parser = add_parse_option(parser)
    args = parser.parse_args()

    # check output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_train and not args.do_eval and not args.do_analyze and not args.do_reinforce:
        raise ValueError("At least one of `do_train`, `do_eval`, `do_analyze`, `do_reinforce` must be True.")

    # Logger
    tb_file_name = args.output_dir.split('/')[-1]
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, "%s.txt"%(tb_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

    # Tensorboard logging
    if not args.do_not_use_tensorboard:
        summary_writer = SummaryWriter("./%s/%s" % (args.tf_dir, tb_file_name))
    else:
        summary_writer = None

    # CUDA setup
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Set hyper-parameters
    lambdas = {'loss_bs':1.0, 'loss_domain':args.lambda_domain, 'loss_act':args.lambda_act}

    ###############################################################################
    # Load data
    ###############################################################################

    # Get Processor
    processor = Processor(args)
    entire_labels = processor.get_labels()
    label_list = entire_labels[0]
    save_configure(args, entire_labels, processor)

    # tokenizer
    vocab_dir = os.path.join(args.bert_dir, '%s-vocab.txt' % args.bert_model)
    if not os.path.exists(vocab_dir):
        raise ValueError("Can't find %s " % vocab_dir)
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=args.do_lower_case)

    num_train_steps = None
    accumulation = False
    
    def get_tensor(data_type, examples):
        # data_type = 'train' or 'dev'
        tensor_dir = os.path.join(args.data_dir, '%s_tensor.json' % data_type)
        if os.path.exists(tensor_dir):
            all_data = torch.load(tensor_dir)
            all_input_ids, all_input_len, all_label_ids, all_output_ids, all_db_feat = all_data
        else:
            all_input_ids, all_input_len, all_label_ids, all_output_ids, all_db_feat = processor.convert_examples_to_features(
                examples, entire_labels, args.max_seq_length, tokenizer, args.max_turn_length)
            torch.save([all_input_ids, all_input_len, all_label_ids, all_output_ids, all_db_feat], tensor_dir)

        num_train_steps = None
        if data_type == 'train':
            num_train_batches = all_input_ids.size(0)
            num_train_steps = int(num_train_batches / args.train_batch_size * args.num_train_epochs)

        all_input_ids, all_input_len = all_input_ids.to(device), all_input_len.to(device)
        for i, label_id in enumerate(all_label_ids):
            all_label_ids[i] = label_id.to(device)

        data = TensorDataset(all_input_ids, all_input_len, all_label_ids[0], all_label_ids[1], all_label_ids[2], all_output_ids, all_db_feat)
        return data, num_train_steps

    if args.do_train or args.do_reinforce:
        train_examples = processor.get_train_examples(args.data_dir, accumulation=accumulation)
        dev_examples = processor.get_dev_examples(args.data_dir, accumulation=accumulation)
        
        # Train utterances
        train_data, num_train_steps = get_tensor('train', train_examples)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        ## Dev utterances
        dev_data, _ = get_tensor('dev', dev_examples)
        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    # Prepare model
    model = SUMBTLaRL(args, processor, entire_labels, device)
    if args.fp16:
        model.half()
    model.to(device)
        
    # Model initialize
    if args.ptr_sumbt_dir != '' or args.ptr_larl_dir != '':
        # Load model if pre-trained models exist and save it to the output dir
        if args.ptr_sumbt_dir != '':
            logger.info("Initialize SUMBT from %s" % args.ptr_sumbt_dir)
            model.initialize_sumbt(args.ptr_sumbt_dir)
        if args.ptr_larl_dir != '':
            logger.info("Initialize LaRL from %s" % args.ptr_larl_dir)
            model.initialize_larl(args.ptr_larl_dir)
        
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), output_model_file)

    elif args.ptr_model_dir != '':
        model = load_model(model, args.ptr_model_dir, n_gpu)
    else:
        # Initialize SUMBT Slot and Slot-values
        ## Get slot-value embeddings
        label_token_ids, label_len = [], []
        for labels in label_list:
            token_ids, lens = get_label_embedding(labels, args.max_label_length, tokenizer, device)
            label_token_ids.append(token_ids)
            label_len.append(lens)

        ## Get domain-slot-type embeddings
        slot_token_ids, _ = \
            get_label_embedding(processor.target_slot, args.max_label_length, tokenizer, device)

        ## Initialize slot and value embeddings
        model.initialize_slot_value_lookup(label_token_ids, slot_token_ids)

    # Data parallelize when use multi-gpus
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train or args.do_reinforce:
        def get_optimizer_grouped_parameters(model):
            param_optimizer_dst = [(n, p) for n, p in model.dst.named_parameters() if p.requires_grad]
            param_optimizer_policy = [(n, p) for n, p in model.policy.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer_dst if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0002,
                 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer_dst if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer_policy if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0002,
                 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer_policy if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.learning_rate},
            ]
            return optimizer_grouped_parameters

        if n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model.module)

        t_total = num_train_steps

        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)
        logger.info(optimizer)


    ###############################################################################
    # define learning functions
    ###############################################################################

    def forward_single_batch(model, batch, n_gpu=1):
        # Forward
        output = model(batch, n_gpu)
        loss = 0.0
        for k, v in output.items():
            if ('loss' in k):
                hyper_param = lambdas[k] if k in lambdas else 1.0
                if n_gpu == 1:
                    loss += hyper_param * v
                else:
                    # average out to multi-gpus
                    loss += hyper_param * v.mean()
        return output, loss
        
    def train_single_batch(model, batch, optimizer, global_step, n_gpu=1):
        batch = tuple(t.to(device) for t in batch)
        output, loss = forward_single_batch(model, batch, n_gpu)

        # Backward
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        # Update
        # modify learning rate with special warm up BERT uses
        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step
        if summary_writer is not None:
            summary_writer.add_scalar("Epoch/LearningRate", lr_this_step, global_step)
            summary_writer.add_scalar("Epoch", epoch, global_step)

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
            
        return output, loss, global_step
    
    def validate(model, dataloader):
        model.eval()
        loss_manager = LossManager(processor)
        n_gpu = 1
        for batch in tqdm(dataloader, desc="RL Validation"):
            batch = tuple(t.to(device) for t in batch)

            # Forward
            with torch.no_grad():
                output, loss = forward_single_batch(model, batch, n_gpu)
            
            num_valid_turn = torch.sum(batch[0][:,:,0].view(-1) > 0, 0).item()
            if 'pred_slot' in output:
                loss_manager.eval_all_accs(output['pred_slot'], output['pred_domain'], output['pred_act'], batch[2:5])
            loss_manager.add_num_data(num_valid_turn)
            loss_manager.add_total_loss(loss, num_valid_turn)
            loss_manager.add_loss(output, num_valid_turn)
                        
        total_loss = loss_manager.get_total_loss()
        losses = loss_manager.get_losses()
        ppl = math.exp(losses['nll']) if 'nll' in losses else 0
        
        return total_loss, losses, ppl

    def gen_task_evaluation():
        evaluator = MultiWozEvaluator('SysWoz', args.delex_data_dir)
        eval_keys = get_key_list(eval_examples)
        
        eval_data, _ = get_tensor('eval', eval_examples)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        assert n_gpu == 1

        with open(os.path.join(args.output_dir, "test_file.txt"), 'w') as f:
            success, match, bleu, f1 = task_generate(args.eval_level, model, eval_dataloader, eval_keys, args, evaluator, 
                                        bert_tokenizer=tokenizer,
                                        dec_vocab=processor.vocab,
                                        device=device, num_batch=None, dest_f=f)
        with open(os.path.join(args.output_dir, "%s.txt" % 'eval_all_accuracies'), 'a') as f:
            logger.info("Successs : %.3e | Match : %.3e | BLEU : %.3e | F1 : %.3e" % (success, match, bleu, f1))
            f.write("Successs : %.3e | Match : %.3e | BLEU : %.3e | F1 : %.3e" % (success, match, bleu, f1))

    
    ###############################################################################
    # Training
    ###############################################################################

    # Training
    if args.do_train:
        logger.info("Training...")

        belief_state = (init_state())['belief_state']

        global_step = 0
        last_update = None
        best_loss = None
        loss_manager = LossManager(processor)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # Train
            model.train()
            loss_manager.init()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                
                batch = tuple(t.to(device) for t in batch)
                output, loss, global_step = train_single_batch(model, batch, optimizer, global_step, n_gpu)
                
                # tensorboard logging
                if summary_writer is not None:
                    tensorboard_logging(summary_writer, 'Train', global_step, 
                                        total_loss=loss, losses=output, stats=None)
                                
            # Perform evaluation on validation dataset
            model.eval()
            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
                batch = tuple(t.to(device) for t in batch)
                
                # Forward
                with torch.no_grad():
                    output, loss = forward_single_batch(model, batch, n_gpu)
                    
                num_valid_turn = torch.sum(batch[0][:,:,0].view(-1) > 0, 0).item()
                if 'pred_slot' in output:
                    loss_manager.eval_all_accs(output['pred_slot'], output['pred_domain'], output['pred_act'], batch[2:5])
                loss_manager.add_num_data(num_valid_turn)
                loss_manager.add_total_loss(loss, num_valid_turn)
                loss_manager.add_loss(output, num_valid_turn)
                
                # Generate
                if step == 0 and not args.pretrain_sumbt:
                    labels = batch[5]
                    num_turn = labels.size(1)                    
                    context, true_labels, pred_labels, state_list = generate(model, batch, belief_state, gen_type=args.gen_type)
                    
                    for did in range(2): #range(num_dialog):            
                        for tid in range(num_turn):
                            # check padded turn
                            if context[did,tid,0] == 0:
                                continue
                            
                            de_tknize = lambda x: ' '.join(x)
                            ctx_str = get_bert_sent(tokenizer, context[did, tid, :])
                            true_str = get_sent(processor.vocab, de_tknize, true_labels[did, tid, :])
                            pred_str = get_sent(processor.vocab, de_tknize, pred_labels[did, tid, :])
                        
                            prev_ctx = 'Source context: %s' % ctx_str
                            
                            logger.info(prev_ctx)
                            logger.info('True: {}'.format(true_str, ))
                            logger.info('Pred: {}'.format(pred_str, ))
                            logger.info('-' * 40)

            total_loss = loss_manager.get_total_loss()
            losses = loss_manager.get_losses()
            ppl = math.exp(losses['nll']) if not args.pretrain_sumbt else 0

            # tensorboard logging
            if summary_writer is not None:
                tensorboard_logging(summary_writer, 'Valid', global_step, total_loss, losses, stats=None)

                precision_domain, precision_act, _, _ = loss_manager.get_precision()
                summary_writer.add_scalar("Valid/Precision_Domain", precision_domain, global_step)
                summary_writer.add_scalar("Valid/Precision_Act", precision_act, global_step)

            msg = '\t Validation: %3e | %3e | ' %(total_loss, ppl)
            msg += ' '.join(['%s %.3e' % (k, x) for k, x in losses.items()])
            logger.info(msg)

            dev_loss = round(total_loss, 6)
            if last_update is None or dev_loss < best_loss:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                if args.do_train:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)

                last_update = epoch
                best_loss = dev_loss
                best_acc = ppl

                logger.info("*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation PPL=%.6f ***" %
                            (last_update, best_loss, best_acc))

            else:
                logger.info("*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation PPL=%.6f  ***" % (epoch, dev_loss, ppl))

            if last_update + args.patience <= epoch:
                break        

    ###############################################################################
    # Evaluation
    ###############################################################################    

    # Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
        
        eval_data, _ = get_tensor('eval', eval_examples)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model = load_model(model, args.output_dir, n_gpu)
        model = model.to(device)
        model.eval()
        loss_manager = LossManager(processor)
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            
            # Forward
            with torch.no_grad():
                output, loss = forward_single_batch(model, batch, n_gpu)
            
            num_valid_turn = torch.sum(batch[0][:,:,0].view(-1) > 0, 0).item()
            if 'pred_slot' in output:
                loss_manager.eval_all_accs(output['pred_slot'], output['pred_domain'], output['pred_act'], batch[2:5])
            loss_manager.add_num_data(num_valid_turn)
            loss_manager.add_total_loss(loss, num_valid_turn)
            loss_manager.add_loss(output, num_valid_turn)
            
        out_file_name = 'eval_all_accuracies'
        with open(os.path.join(args.output_dir, "%s.txt" % out_file_name), 'w') as f:
            f.write(loss_manager.print_result())
            
        ###############################################################################
        # Generation and Task Evaluation
        ###############################################################################   

        if not args.pretrain_sumbt:
            gen_task_evaluation()
    
    ###############################################################################
    # Analysis
    ###############################################################################       
    if args.do_analyze and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # using vis_act_dist.ipynb
        eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
        
        eval_data, _ = get_tensor('eval', eval_examples)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model = load_model(model, args.output_dir, n_gpu)
        model = model.to(device)
        model.eval()
        loss_manager = LossManager(processor)
        
        eval_data, _ = get_tensor('eval', eval_examples)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
        #TODO: from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.enc2dec.decoders import GEN

        sample_z = torch.zeros(args.y_size, args.k_size).to(device)
        log_qz = torch.zeros(args.y_size, args.k_size).to(device)

        #'output_bs', 'loss_bs', 'acc', 'pred_slot', 'loss_domain', 'acc_domain', 'pred_domain', 'loss_act', 'acc_act', 'pred_act', 'attention_score', 'sequence', 'sample_z', 'log_qy'

        #for batch in tqdm(eval_dataloader, desc="Analyze"):
        batch = eval_dataloader[0]
        batch = tuple(t.to(device) for t in batch)
        
        label_ids = batch[1]
        mask = (label_ids == 0)[:,:,0] # batch, turn 
        num_data = ( mask == 0).sum()
        nbatch = mask.size(0)
        nturn = mask.size(1)

        mask = mask.unsqueeze(-1).unsqueeze(-1).expand(16,18,args.y_size, args.k_size)

        # Forward
        with torch.no_grad():
            outputs = model(batch, n_gpu=1, mode=GEN, gen_type=args.gen_type)
        
        # Add info
        sample_z += outputs['sample_z'].view(nbatch, nturn, args.y_size, args.k_size).masked_fill(mask,0).sum(0).sum(0)/num_data
        log_qz += torch.nn.functional.softmax(outputs['log_qy'],-1).view(nbatch, nturn, args.y_size, args.k_size).masked_fill(mask,0).sum(0).sum(0)/num_data
        attention_score = torch.cat(outputs['attention_score'],1).view(nbatch, nturn, -1, args.y_size)
        
        # convert index to senetences
        #TODO: from convlab.modules.e2e.multiwoz.SUMBT_LaRL.utils.processor import get_sent, get_bert_sent

        context = batch[0].cpu().numpy()
        true_labels = batch[5][:, :, 1:].cpu().numpy() # (batch_size(num_dialog), num_turn, output_seq_len)

        pred_labels = torch.cat(outputs['sequence'], dim=-1) #(ds*ts, output_seq_len)
        pred_labels = pred_labels.view(nbatch, nturn, -1).long()
        pred_labels = pred_labels.cpu().numpy()

        de_tknize = de_tknize = lambda x: ' '.join(x)
        pred_str = []
        true_str = []
        cont_str = []
        for i, dialog in enumerate(pred_labels):
            pred_t = []
            true_t = []
            cont_t = []
            for j, _ in enumerate(dialog):
                pred_t.append(get_sent(processor.vocab, de_tknize, pred_labels[i, j, :]))
                true_t.append(get_sent(processor.vocab, de_tknize, true_labels[i, j, :]))
                cont_t.append(get_bert_sent(tokenizer, context[i,j, :]))
                
            pred_str.append(pred_t)
            true_str.append(true_t)
            cont_str.append(cont_t)

        # save
        torch.save({'context':cont_str, 'true':true_str, 'pred':pred_str, 
            'sample_z':sample_z, 'log_qz':log_qz, 'attention_score':attention_score}, 'analyze_data.bin')

        pdb.set_trace()

    ###############################################################################
    # REINFORCE
    ###############################################################################    
    
    if args.do_reinforce and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # make exp directory
        start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))        
        output_rl_dir = os.path.join(args.output_dir, 'rl-'+start_time)
        os.makedirs(output_rl_dir, exist_ok=True)

        # define logger
        tb_file_name = args.output_dir.split('/')[-1]
        fileHandler = logging.FileHandler(os.path.join(output_rl_dir, "log.txt"))
        logger.addHandler(fileHandler)
        logger.info(args)
        logger.info('REINFORCE: [START] '+ start_time + ' ='*30)

        # Tensorboard logging
        if not args.do_not_use_tensorboard:
            summary_writer = SummaryWriter("./%s/%s/%s" % (args.tf_dir, tb_file_name, 'rl-'+start_time))
        else:
            summary_writer = None
            
        #args.dropout = 0.0 # TODO? set all dropout layer's prob = 0.0
        
        logger.info("Do we only tune the policy: {}".format(args.tune_pi_only))
        f = open(os.path.join(output_rl_dir, "valid_file.txt"), 'w')
                
        # prepare data and key
        rl_train_data, train_keys = get_data_with_idx(train_examples, train_data, device)
        rl_train_dataloader = DataLoader(rl_train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        #rl_train_dataloader = DataLoader(rl_train_data, sampler=train_sampler, batch_size=1)
        dev_keys = get_key_list(dev_examples)
    
        # Define RL agents and Task
        if args.ptr_model_dir:
            model = load_model(model, args.ptr_model_dir, n_gpu)
        else:
            model = load_model(model, args.output_dir, n_gpu)
        model = model.to(device)
        loss_manager = LossManager(processor)
        
        # prepare agent and evaluator
        agent = OfflineLatentRlAgent(model, train_dataloader, train_keys, args, 
                                     name='System', tune_pi_only=args.tune_pi_only,
                                    bert_tokenizer=tokenizer,
                                    dec_vocab=processor.vocab)
        evaluator = MultiWozEvaluator('SYS_WOZ', args.delex_data_dir)
        
        
        # BEFORE RUN, RECORD INITIAL PERFORMANCE
        agent.model.eval()
        dev_loss, losses, ppl = validate(model, dev_dataloader)
        success, match, bleu, f1 = task_generate(args.eval_level, model, dev_dataloader, dev_keys, args, evaluator, 
                                    bert_tokenizer=tokenizer,
                                    dec_vocab=processor.vocab,
                                    device=device, num_batch=None, dest_f=f, verbose=False)
        stats = {'Success': success, 'Match': match, 'BLEU': bleu, 'F1': f1}        
        # tensorboard logging
        tensorboard_logging(summary_writer, 'Valid', global_step=0, total_loss=dev_loss, losses=losses, stats=stats)
        
        msg = '\t Validation: %3e | %3e | ' %(dev_loss, ppl)
        msg += ' '.join(['%s %.3e' % (k, x) for k, x in stats.items()])
        msg += ' '
        msg += ' '.join(['%s %.3e' % (k, x) for k, x in losses.items()])
        logger.info(msg)
        
        # train epochs:
        num_steps = 0
        rl_global_step = 0
        sl_global_step = 0
        t_total = num_train_steps
        best_loss = None
        best_rewards = None
        last_update = None
        learning_step = args.eval_level

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            agent.model.train()
            
            ##  TODO: episode repeat
            for batch in tqdm(rl_train_dataloader, desc="RL Training"):
                num_steps += 1
                batch = tuple(t.to(device) for t in batch)
    
                agent.model.train()                            
                # reinforcement learning                
                task_report, success, match = agent.run(learning_step, batch, train_keys, evaluator, 
                                                        max_words=args.rl_max_words, temp=args.temperature, use_oracle_bs=args.use_oracle_bs)
                reward = success
                stats = {'Match': np.mean(match), 'Success': np.mean(success)}
                rl_loss = agent.update(reward, stats)
                
                """
                # No warm up when using sgd
                lr_this_step = args.learning_rate * warmup_linear(rl_global_step / t_total, args.warmup_proportion)
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                """
                lr_this_step = agent.args.rl_lr
                rl_global_step += 1
                                
                # supervised learning
                output = None
                loss = None
                if args.sv_train_freq > 0 and rl_global_step % args.sv_train_freq == 0:
                    batch = next(iter(train_dataloader))
                    output, loss, sl_global_step = train_single_batch(model, batch, optimizer, sl_global_step, n_gpu)
                    
                # tensorboard logging
                tensorboard_logging(summary_writer, 'Train', rl_global_step, 
                                    total_loss=loss, losses=output, rl_loss=rl_loss, stats=stats)
                if summary_writer is not None:
                    summary_writer.add_scalar("RL_Epoch/LearningRate", lr_this_step, rl_global_step)
                    summary_writer.add_scalar("RL_Epoch", epoch, rl_global_step)

                if rl_global_step % 200 == 0:
                    # validate: record results
                    agent.model.eval()
                    f.write('Epoch: %d - %d' % (epoch, rl_global_step/t_total))
                    dev_loss, losses, ppl = validate(model, dev_dataloader)
                    success, match, bleu, f1 = task_generate(learning_step, model, dev_dataloader, dev_keys, args, evaluator, 
                                                bert_tokenizer=tokenizer,
                                                dec_vocab=processor.vocab,
                                                device=device, num_batch=None, dest_f=f, verbose=False)
                    stats = {'Success': success, 'Match': match, 'BLEU': bleu, 'F1': f1, 'eval_level': learning_step}
                    
                    # tensorboard logging
                    tensorboard_logging(summary_writer, 'Valid', rl_global_step, total_loss=dev_loss, losses=losses, stats=stats)
                    
                    msg = '\t Validation: %3e | %3e | ' %(dev_loss, ppl)
                    msg += ' '.join(['%s %.3e' % (k, x) for k, x in stats.items()])
                    msg += ' '
                    msg += ' '.join(['%s %.3e' % (k, x) for k, x in losses.items()])
                    logger.info(msg)
                    
                    dev_rewards = match + success
                    if last_update is None or dev_rewards > best_rewards:
                        # Save a trained model
                        output_model_file = os.path.join(output_rl_dir, "pytorch_model.bin")
                        if n_gpu == 1:
                            torch.save(model.state_dict(), output_model_file)
                        else:
                            torch.save(model.module.state_dict(), output_model_file)

                        last_update = rl_global_step
                        best_loss = dev_loss
                        best_acc = ppl
                        best_rewards = dev_rewards
                        logger.info("*** Model Updated: Step=%d, Validation Rewards %.6f Success %.6f Match %.6f Validation Loss=%.6f, Validation PPL=%.6f ***" %
                                    (last_update, best_rewards, success,  match,best_loss, best_acc))

                    else:
                        logger.info("*** Model NOT Updated: Step=%d, Validation Rewards %.6f Success  %.6f Match %.6f Validation Loss=%.6f, Validation PPL=%.6f  ***" %
                                    (last_update, dev_rewards, success, match, dev_loss, ppl))

                    if last_update + args.patience * 200 <= rl_global_step:
                        break
    
        # evaluate
        gen_task_evaluation()
        
        end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        logger.info('[END] ' + end_time + ' ='*30)


def load_model(model, output_dir, n_gpu):
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    # in the case that slot and values are different between the training and evaluation
    ptr_model = torch.load(output_model_file)

    if n_gpu == 1:
        model.load_state_dict(ptr_model)
    else:
        raise ValueError("Evaluate using only one device!")
    
    return model

def get_data_with_idx(examples, data, device):
    # get key list
    keys = get_key_list(examples)
    num_data = len(keys)
    # Append data idx at trian_data       
    data = TensorDataset(torch.LongTensor(range(num_data)).to(device), *data.tensors)
    return data, keys
    
def tensorboard_logging(summary_writer, type, global_step, total_loss, losses, rl_loss=None, stats=None):
    logging=['pi_kl', 'diversity', 'nll', 'b_pr', 'mi']
    if summary_writer is not None:
        if total_loss is not None:
            summary_writer.add_scalar("%s/Loss"%type, total_loss, global_step)
        if rl_loss is not None:
            summary_writer.add_scalar("%s/RL_Loss"%type, rl_loss, global_step)
        if stats is not None:
            for k, v in stats.items():
                summary_writer.add_scalar("%s/%s"%(type, k), v, global_step)
        if losses is not None:
            for k, v in losses.items():
                if ('loss' in k) or ('acc' in k) or (k in logging):
                    summary_writer.add_scalar("%s/%s"%(type, k), v, global_step)
             

if __name__ == "__main__":
    main()