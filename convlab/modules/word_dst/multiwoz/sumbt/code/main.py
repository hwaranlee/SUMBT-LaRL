###
# Unified SUMBT


import csv
import os
import logging
import argparse
import random
import collections
from tqdm import tqdm, trange
import json

import pdb

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from tensorboardX import SummaryWriter

from processor import Processor, convert_examples_to_features, get_label_embedding

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_dir", default='/home/.pytorch_pretrained_bert',
                        type=str, required=False,
                        help="The directory of the pretrained BERT model")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train: bert-gru-sumbt, bert-lstm-sumbt"
                             "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--target_slot",
                        default='all',
                        type=str,
                        required=True,
                        help="Target slot idx to train model. ex. 'all', '0:1:2', or an excluding slot name 'attraction'" )
    parser.add_argument("--tf_dir",
                        default='tensorboard',
                        type=str,
                        required=False,
                        help="Tensorboard directory")
    parser.add_argument("--nbt",
                        default='rnn',
                        type=str,
                        required=True,
                        help="nbt type: rnn or transformer" )
    parser.add_argument("--fix_utterance_encoder",
                        action='store_true',
                        help="Do not train BERT utterance encoder")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_label_length",
                        default=32,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_turn_length",
                        default=22,
                        type=int,
                        help="The maximum total input turn length. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=100,
                        help="hidden dimension used in belief tracker")
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=1,
                        help="number of RNN layers")
    parser.add_argument('--zero_init_rnn',
                        action='store_true',
                        help="set initial hidden of rnns zero")
    parser.add_argument('--attn_head',
                        type=int,
                        default=4,
                        help="the number of heads in multi-headed attention")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--distance_metric",
                        type=str,
                        default="cosine",
                        help="The metric for distance between label embeddings: cosine, euclidean.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total dialog batch size for training.")
    parser.add_argument("--dev_batch_size",
                        default=1,
                        type=int,
                        help="Total dialog batch size for validation.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total dialog batch size for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for BertAdam.")
    parser.add_argument("--lambda_req",
                        default=10,
                        type=float,
                        help="Hyperparameter for loss_req")
    parser.add_argument("--lambda_gen",
                        default=10,
                        type=float,
                        help="Hyperparameter for loss_req")
    parser.add_argument("--req_pos_weight",
                        default=10,
                        type=float,
                        help="Positive weight of BCE Loss")
    parser.add_argument("--gen_pos_weight",
                        default=1,
                        type=float,
                        help="Positive weight of BCE Loss")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--patience",
                        default=10.0,
                        type=float,
                        help="The number of epochs to allow no further improvement.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--do_not_use_tensorboard",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    args = parser.parse_args()

    # check output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_train and not args.do_eval and not args.do_analyze:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Tensorboard logging
    if not args.do_not_use_tensorboard:
        tb_file_name = args.output_dir.split('/')[1]
        summary_writer = SummaryWriter("./%s/%s" % (args.tf_dir, tb_file_name))
    else:
        summary_writer = None

    # Logger
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, "%s.txt"%(tb_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

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

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    ###############################################################################
    # Load data
    ###############################################################################

    # Get Processor
    processor = Processor(args)
    entire_labels = processor.get_labels()
    label_list, req_list, gen_list = entire_labels
    save_configure(args, entire_labels, processor.ontology)

    # tokenizer
    vocab_dir = os.path.join(args.bert_dir, '%s-vocab.txt' % args.bert_model)
    if not os.path.exists(vocab_dir):
        raise ValueError("Can't find %s " % vocab_dir)
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=args.do_lower_case)

    num_train_steps = None
    accumulation = False

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, accumulation=accumulation)
        dev_examples = processor.get_dev_examples(args.data_dir, accumulation=accumulation)

        ## Training utterances
        tensor_dir = os.path.join(args.data_dir, 'train_tensor.json')
        if os.path.exists(tensor_dir):
            all_data = torch.load(tensor_dir)
            all_input_ids, all_input_len, all_label_ids = all_data
        else:
            all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
                train_examples, entire_labels, args.max_seq_length, tokenizer, args.max_turn_length)
            torch.save([all_input_ids, all_input_len, all_label_ids ], tensor_dir)

        num_train_batches = all_input_ids.size(0)
        num_train_steps = int(num_train_batches / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids, all_input_len = all_input_ids.to(device), all_input_len.to(device)
        for i, label_id in enumerate(all_label_ids):
            all_label_ids[i] = label_id.to(device)

        train_data = TensorDataset(all_input_ids, all_input_len, all_label_ids[0], all_label_ids[1], all_label_ids[2])
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        ## Dev utterances
        tensor_dir = os.path.join(args.data_dir, 'dev_tensor.json')
        if os.path.exists(tensor_dir):
            all_data = torch.load(tensor_dir)
            all_input_ids_dev, all_input_len_dev, all_label_ids_dev = all_data
        else:
            all_input_ids_dev, all_input_len_dev, all_label_ids_dev = convert_examples_to_features(
                dev_examples, entire_labels, args.max_seq_length, tokenizer, args.max_turn_length)
            torch.save([all_input_ids_dev, all_input_len_dev, all_label_ids_dev ], tensor_dir)

        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)

        all_input_ids_dev, all_input_len_dev = all_input_ids_dev.to(device), all_input_len_dev.to(device)
        for i, label_id in enumerate(all_label_ids):
            all_label_ids[i] = label_id.to(device)

        dev_data = TensorDataset(all_input_ids_dev, all_input_len_dev, all_label_ids_dev[0], all_label_ids_dev[1], all_label_ids_dev[2])
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

    logger.info("Loaded data!")

    ###############################################################################
    # Build the models
    ###############################################################################

    # Prepare model
    if args.nbt =='rnn':
        from sumbt_unified import BeliefTracker
    elif args.nbt == 'transformer':
        raise ValueError('transformer not is not working now')
        from BeliefTrackerSlotQueryMultiSlotTransformer import BeliefTracker
    else:
        raise ValueError('nbt type should be either rnn or transformer')

    model = BeliefTracker(args, entire_labels, device)
    if args.fp16:
        model.half()
    model.to(device)

    ## Get slot-value embeddings
    label_token_ids, label_len = [], []
    for labels in label_list:
        token_ids, lens = get_label_embedding(labels, args.max_label_length, tokenizer, device)
        label_token_ids.append(token_ids)
        label_len.append(lens)

    ## Get domain-slot-type embeddings
    slot_token_ids, _ = \
        get_label_embedding(processor.target_slot, args.max_label_length, tokenizer, device)

    ## Get domain-slot-type embeddings for Request
    req_slot_token_ids, _ = \
        get_label_embedding(processor.ontology_request, args.max_label_length, tokenizer, device)

    ## Initialize slot and value embeddings
    model.initialize_slot_value_lookup(label_token_ids, slot_token_ids, req_slot_token_ids)

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
    if args.do_train:
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
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
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0
        last_update = None
        best_loss = None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len = batch[0], batch[1]
                label_ids = batch[2:]

                # Forward
                if n_gpu == 1:
                    loss_bs, loss_slot, acc, acc_slot, _ , loss_req, acc_req, _, loss_gen, acc_gen, _ = \
                        model(input_ids, input_len, label_ids, n_gpu)
                    loss = loss_bs + args.lambda_req * loss_req + args.lambda_gen * loss_gen

                else:
                    loss, _, acc, acc_slot, _ = model(input_ids, input_len, label_ids, n_gpu)

                    # average to multi-gpus
                    loss = loss.mean()
                    acc = acc.mean()
                    acc_slot = acc_slot.mean(0)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # tensrboard logging
                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch", epoch, global_step)
                    summary_writer.add_scalar("Train/Loss", loss, global_step)
                    summary_writer.add_scalar("Train/Loss_BS", loss_bs, global_step)
                    summary_writer.add_scalar("Train/JointAcc", acc, global_step)
                    summary_writer.add_scalar("Train/Loss_REQ", loss_req, global_step)
                    summary_writer.add_scalar("Train/Acc_REQ", acc_req, global_step)
                    summary_writer.add_scalar("Train/Loss_GEN", loss_gen, global_step)
                    summary_writer.add_scalar("Train/Acc_GEN", acc_gen, global_step)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify lealrning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    if summary_writer is not None:
                        summary_writer.add_scalar("Train/LearningRate", lr_this_step, global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Perform evaluation on validation dataset
            model.eval()
            dev_loss = 0
            dev_losses = [0, 0, 0]
            dev_acc = [0, 0, 0]
            dev_loss_slot, dev_acc_slot = None, None
            nb_dev_examples, nb_dev_steps = 0, 0

            accuracies = {'joint7': 0, 'slot7': 0, 'joint5': 0, 'slot5': 0, 'joint_rest': 0, 'slot_rest': 0,
                          'num_turn': 0, 'num_slot7': 0, 'num_slot5': 0, 'num_slot_rest': 0,
                          'req_tp': 0, 'req_tn': 0, 'req_pos': 0, 'req_neg': 0,
                          'gen_tp': 0, 'gen_tn': 0, 'gen_pos': 0, 'gen_neg': 0}

            for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_len = batch[0], batch[1]
                label_ids = batch[2:]

                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)

                with torch.no_grad():
                    if n_gpu == 1:
                        loss_bs, loss_slot, acc, acc_slot, pred_slot, loss_req, acc_req, pred_req, loss_gen, acc_gen, pred_gen = \
                            model(input_ids, input_len, label_ids, n_gpu)
                        loss = loss_bs + args.lambda_req * loss_req + args.lambda_gen * loss_gen

                    else:
                        loss, _, acc, acc_slot, _= model(input_ids, input_len, label_ids, n_gpu)

                        # average to multi-gpus
                        loss = loss.mean()
                        acc = acc.mean()
                        acc_slot = acc_slot.mean(0)

                accuracies = eval_all_accs(pred_slot, pred_req, pred_gen, label_ids, accuracies)

                num_valid_turn = torch.sum(label_ids[0][:,:,0].view(-1) > -1, 0).item()
                for i, val in enumerate(zip([loss_bs, loss_req, loss_gen], [acc, acc_req, acc_gen])):
                    dev_losses[i] += val[0].item() * num_valid_turn
                    dev_acc[i] += val[1].item() * num_valid_turn

                """
                if n_gpu == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [ l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn
                """

                dev_loss += loss.item() * num_valid_turn
                nb_dev_examples += num_valid_turn

            dev_loss = dev_loss / nb_dev_examples

            for i, loss in enumerate(dev_losses):
                dev_losses[i] = dev_losses[i] /nb_dev_examples
            for i, acc in enumerate(dev_acc):
                dev_acc[i] = dev_acc[i] / nb_dev_examples

            tp = accuracies['req_tp'].item()
            pos = accuracies['req_pos'].item()
            precision_req = tp/pos
            tp = accuracies['gen_tp'].item()
            pos = accuracies['gen_pos'].item()
            precision_gen = tp / pos

            # tensorboard logging
            if summary_writer is not None:
                summary_writer.add_scalar("Valid/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Valid/Loss_BS", dev_losses[0], global_step)
                summary_writer.add_scalar("Valid/JointAcc", dev_acc[0], global_step)
                summary_writer.add_scalar("Valid/Loss_REQ", dev_losses[1], global_step)
                summary_writer.add_scalar("Valid/Acc_REQ", dev_acc[1], global_step)
                summary_writer.add_scalar("Valid/Precision_REQ", precision_req, global_step)
                summary_writer.add_scalar("Valid/Loss_GEN", dev_losses[2], global_step)
                summary_writer.add_scalar("Valid/Acc_GEN", dev_acc[2], global_step)
                summary_writer.add_scalar("Valid/Precision_GEN", precision_gen, global_step)

            msg = '\t Validation: '
            msg += ' '.join(['%.3e' % x for x in dev_losses])
            msg += ' | '
            msg += ' '.join(['%.3e' % x for x in dev_acc])
            logger.info(msg)

            dev_loss = round(dev_loss, 6)
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
                best_acc = dev_acc

                logger.info("*** Model Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f ***" %
                            (last_update, best_loss, best_acc[0]))

            else:
                logger.info("*** Model NOT Updated: Epoch=%d, Validation Loss=%.6f, Validation Acc=%.6f  ***" % (epoch, dev_loss, dev_acc[0]))

            if last_update + args.patience <= epoch:
                break


    ###############################################################################
    # Evaluation
    ###############################################################################
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    model = BeliefTracker(args, entire_labels, device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # in the case that slot and values are different between the training and evaluation
    ptr_model = torch.load(output_model_file)

    if n_gpu == 1:
        state = model.state_dict()
        state.update(ptr_model)
        model.load_state_dict(state)
    else:
        print("Evaluate using only one device!")
        model.module.load_state_dict(ptr_model)

    model.to(device)

    if n_gpu != 1:
        raise ValueError("Evaluation requires 1 GPU")

    # Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
        ## Test utterances
        tensor_dir = os.path.join(args.data_dir, 'test_tensor.json')
        if os.path.exists(tensor_dir):
            all_data = torch.load(tensor_dir)
            all_input_ids, all_input_len, all_label_ids = all_data
        else:
            all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
                eval_examples, entire_labels, args.max_seq_length, tokenizer, args.max_turn_length)
            torch.save([all_input_ids, all_input_len, all_label_ids], tensor_dir)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids, all_input_len = all_input_ids.to(device), all_input_len.to(device)
        for i, label_id in enumerate(all_label_ids):
            all_label_ids[i] = label_id.to(device)

        eval_data = TensorDataset(all_input_ids, all_input_len, all_label_ids[0], all_label_ids[1], all_label_ids[2])

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        eval_losses = [0, 0, 0]
        eval_acc = [0, 0, 0]
        eval_loss_slot, eval_acc_slot = None, None
        nb_eval_steps, nb_eval_examples = 0, 0

        accuracies = {'joint7':0, 'slot7':0, 'joint5':0, 'slot5':0, 'joint_rest':0, 'slot_rest':0,
                      'num_turn':0, 'num_slot7':0, 'num_slot5':0, 'num_slot_rest':0,
                      'req_tp':0, 'req_tn':0, 'req_pos':0, 'req_neg':0,
                      'gen_tp': 0, 'gen_tn': 0, 'gen_pos': 0, 'gen_neg': 0}

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids, input_len = batch[0], batch[1]
            label_ids = batch[2:]

            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(0)
                input_len = input_len.unsqueeze(0)
                for i in range(len(label_ids)):
                    label_ids[i] = label_ids[i].unsuqeeze(0)

            with torch.no_grad():
                loss_bs, loss_slot, acc, acc_slot, pred_slot, loss_req, acc_req, pred_req, loss_gen, acc_gen, pred_gen= \
                        model(input_ids, input_len, label_ids, n_gpu)
                loss = loss_bs + args.lambda_req * loss_req + args.lambda_gen * loss_gen

            accuracies = eval_all_accs(pred_slot, pred_req, pred_gen, label_ids, accuracies)

            nb_eval_ex = (label_ids[0][:,:,0].view(-1) != -1).sum().item()
            nb_eval_examples += nb_eval_ex
            nb_eval_steps += 1

            eval_loss += loss.item() * nb_eval_ex
            if eval_loss_slot is None:
                eval_loss_slot = [ l * nb_eval_ex for l in loss_slot]
                eval_acc_slot = acc_slot * nb_eval_ex
            else:
                for i, l in enumerate(loss_slot):
                    eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
                eval_acc_slot += acc_slot * nb_eval_ex

            for i, val in enumerate(zip([loss_bs, loss_req, loss_gen], [acc, acc_req, acc_gen])):
                eval_losses[i] += val[0].item() * nb_eval_ex
                eval_acc[i] += val[1].item() * nb_eval_ex

        eval_loss = eval_loss / nb_eval_examples
        eval_acc_slot = eval_acc_slot / nb_eval_examples

        for i, loss in enumerate(eval_losses):
            eval_losses[i] = loss / nb_eval_examples
        for i, acc in enumerate(eval_acc):
            eval_acc[i] = acc / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_loss_bs': eval_losses[0],
                  'eval_accuracy': eval_acc[0],
                  'eval_loss_slot':'\t'.join([ str(val/ nb_eval_examples) for val in eval_loss_slot]),
                  'eval_acc_slot':'\t'.join([ str((val).item()) for val in eval_acc_slot]),
                  'eval_loss_req': eval_losses[1],
                  'eval_acc_req': eval_acc[1],
                  'eval_loss_gen': eval_losses[2],
                  'eval_acc_gen': eval_acc[2]
                  }

        out_file_name = 'eval_results'
        if args.target_slot=='all':
            out_file_name += '_all'
        output_eval_file = os.path.join(args.output_dir, "%s.txt" % out_file_name)

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        out_file_name = 'eval_all_accuracies'
        with open(os.path.join(args.output_dir, "%s.txt" % out_file_name), 'w') as f:
            f.write('joint acc (7 domain) : slot acc (7 domain) : joint acc (5 domain): slot acc (5 domain): joint restaurant : slot acc restaurant \n')
            f.write('%.5f : %.5f : %.5f : %.5f : %.5f : %.5f \n' % (
                (accuracies['joint7']/accuracies['num_turn']).item(),
                (accuracies['slot7']/accuracies['num_slot7']).item(),
                (accuracies['joint5']/accuracies['num_turn']).item(),
                (accuracies['slot5'] / accuracies['num_slot5']).item(),
                (accuracies['joint_rest']/accuracies['num_turn']).item(),
                (accuracies['slot_rest'] / accuracies['num_slot_rest']).item()
            ))

            tp = accuracies['req_tp'].item()
            tn = accuracies['req_tn'].item()
            pos = accuracies['req_pos'].item()
            neg = accuracies['req_neg'].item()

            f.write('REQUEST ACC : REQUEST PRECISION : TP : TN : POS : NEG \n')
            f.write('%.5f : %.5f : %d : %d : %d : %d \n' %(
                    (tp+tn)/(pos+neg), tp/pos, tp, tn, pos, neg))

            tp = accuracies['gen_tp'].item()
            tn = accuracies['gen_tn'].item()
            pos = accuracies['gen_pos'].item()
            neg = accuracies['gen_neg'].item()

            f.write('GENERAL ACC : GENERAL PRECISION : TP : TN : POS : NEG \n')
            f.write('%.5f : %.5f : %d : %d : %d : %d \n' % (
                (tp + tn) / (pos + neg), tp / pos, tp, tn, pos, neg))


def eval_all_accs(pred_slot, pred_req, pred_gen, labels, accuracies):

    def _eval_acc(_pred_slot, _labels):
        slot_dim = _labels.size(-1)
        accuracy = (_pred_slot == _labels).view(-1, slot_dim)
        num_turn = torch.sum(_labels[:, :, 0].view(-1) > -1, 0).float()
        num_data = torch.sum(_labels > -1).float()
        # joint accuracy
        joint_acc = sum(torch.sum(accuracy, 1) / slot_dim).float()
        # slot accuracy
        slot_acc = torch.sum(accuracy).float()
        return joint_acc, slot_acc, num_turn, num_data

    label_bs, label_req, label_gen = labels

    # 7 domains
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot, label_bs)
    accuracies['joint7'] += joint_acc
    accuracies['slot7'] += slot_acc
    accuracies['num_turn'] += num_turn
    accuracies['num_slot7'] += num_data

    # restaurant domain
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot[:,:,20:27], label_bs[:,:,20:27])
    accuracies['joint_rest'] += joint_acc
    accuracies['slot_rest'] += slot_acc
    accuracies['num_slot_rest'] += num_data

    # 5 domains (excluding bus and hotel domain)
    pred_slot5 = torch.cat((pred_slot[:,:,0:3], pred_slot[:,:,10:]), 2)
    label_slot5 = torch.cat((label_bs[:,:,0:3], label_bs[:,:,10:]), 2)

    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot5, label_slot5)
    accuracies['joint5'] += joint_acc
    accuracies['slot5'] += slot_acc
    accuracies['num_slot5'] += num_data

    # Request
    pos = (label_req == 1)
    neg = (label_req == 0)
    accuracies['req_pos'] += pos.sum() # num of positive samples
    accuracies['req_neg'] += neg.sum() # num of negative samples
    accuracies['req_tp'] += ((pred_req == 1).masked_select(pos)).sum() # true positive
    accuracies['req_tn'] += ((pred_req == 0).masked_select(neg)).sum() # true negative

    # General Act
    pos = (label_gen == 1)
    neg = (label_gen == 0)
    accuracies['gen_pos'] += pos.sum() # num of positive samples
    accuracies['gen_neg'] += neg.sum() # num of negative samples
    accuracies['gen_tp'] += ((pred_gen == 1).masked_select(pos)).sum() # true positive
    accuracies['gen_tn'] += ((pred_gen == 0).masked_select(neg)).sum() # true negative

    return accuracies

def save_configure(args, labels, ontology):
    with open(os.path.join(args.output_dir, "config.json"),'w') as outfile:

        data = { "bert_dir": args.bert_dir,
                 "bert_model": args.bert_model,
                 "do_lower_case": args.do_lower_case,
                 "task_name": args.task_name,
                 "hidden_dim": args.hidden_dim,
                "num_rnn_layers": args.num_rnn_layers,
                "zero_init_rnn": args.zero_init_rnn,
                "max_seq_length": args.max_seq_length,
                "max_label_length": args.max_label_length,
                "attn_head": args.attn_head,
                 "distance_metric": args.distance_metric,
                 "fix_utterance_encoder": args.fix_utterance_encoder,
                 "req_pos_weight": args.req_pos_weight,
                 "gen_pos_weight": args.gen_pos_weight,
                 "labels": labels,
                 "ontology": ontology,
                 }

        json.dump(data, outfile, indent=4)

if __name__ == "__main__":
    main()
