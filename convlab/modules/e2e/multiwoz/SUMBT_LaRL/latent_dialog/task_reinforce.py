import os
import sys
import numpy as np
import torch
from torch import nn
from collections import defaultdict
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.enc2dec.base_modules import summary
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from datetime import datetime
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.utils import get_detokenize
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.corpora import EOS, PAD
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.data_loaders import DealDataLoaders, BeliefDbDataLoaders
from convlab.modules.word_policy.multiwoz.larl.latent_dialog import evaluators
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.record import record, record_task, UniquenessSentMetric, UniquenessWordMetric
import logging

import pdb
import tqdm

class OfflineTaskReinforce(object):
    def __init__(self, agent, dataloaders, args, generate_func, logger):
        pdb.set_trace()
        self.agent = agent
        self.sys_model = agent.model
        self.train_dataloader = dataloaders['train']
        self.valid_dataloader = dataloaders['valid'] if 'valid' in dataloaders else None
        self.eval_dataloader = dataloaders['test'] if 'test' in dataloaders else None
        
        self.args = args
        self.logger = logger 
        
        
        #self.sv_config = sv_config
        #self.sys_model = sys_model
        #self.rl_config = rl_config
        
        # training func for supervised learning
        self.train_func = task_train_single_batch
        self.record_func = record_task
        self.validate_func = validate

        """
        # prepare data loader
        train_dial, val_dial, test_dial = self.corpus.get_corpus()
        self.train_data = BeliefDbDataLoaders('Train', train_dial, self.sv_config)
        self.sl_train_data = BeliefDbDataLoaders('Train', train_dial, self.sv_config)
        self.val_data = BeliefDbDataLoaders('Val', val_dial, self.sv_config)
        self.test_data = BeliefDbDataLoaders('Test', test_dial, self.sv_config)

        # create log files
        if self.rl_config.record_freq > 0:
            self.learning_exp_file = open(os.path.join(self.rl_config.record_path, 'offline-learning.tsv'), 'w')
            self.ppl_val_file = open(os.path.join(self.rl_config.record_path, 'val-ppl.tsv'), 'w')
            self.rl_val_file = open(os.path.join(self.rl_config.record_path, 'val-rl.tsv'), 'w')
            self.ppl_test_file = open(os.path.join(self.rl_config.record_path, 'test-ppl.tsv'), 'w')
            self.rl_test_file = open(os.path.join(self.rl_config.record_path, 'test-rl.tsv'), 'w')
        """
        
        # evaluation
        self.evaluator = evaluators.MultiWozEvaluator('SYS_WOZ')
        self.generate_func = generate_func

    def run(self):
        n = 0
        best_valid_loss = np.inf
        best_rewards = -1 * np.inf
        
        pdb.set_trace()

        # BEFORE RUN, RECORD INITIAL PERFORMANCE
        test_loss = self.validate_func(self.sys_model, self.test_dataloader, self.args, use_py=True)
        t_success, t_match, t_bleu, t_f1 = self.generate_func(self.sys_model, self.test_dataloader, self.args,
                                                              self.evaluator, None, verbose=False)
        """
        self.ppl_test_file.write('{}\t{}\t{}\t{}\n'.format(n, np.exp(test_loss), t_bleu, t_f1))
        self.ppl_test_file.flush()

        self.rl_test_file.write('{}\t{}\t{}\t{}\n'.format(n, (t_success + t_match), t_success, t_match))
        self.rl_test_file.flush()
        """
        self.sys_model.train()
        try:
            for epoch_id in range(self.args.nepoch):
                self.train_dataloader.epoch_init(self.args, shuffle=True, verbose=epoch_id == 0, fix_batch=True)
                while True:
                    if n % self.args.episode_repeat == 0:
                        batch = self.train_dataloader.next_batch()

                    if batch is None:
                        break

                    n += 1
                    if n % 50 == 0:
                        print("Reinforcement Learning {}/{} eposide".format(n, self.train_dataloader.num_batch*self.args.nepoch))
                        self.learning_exp_file.write(
                            '{}\t{}\n'.format(n, np.mean(self.agent.all_rewards[-50:])))
                        self.learning_exp_file.flush()

                    # reinforcement learning
                    # make sure it's the same dialo
                    assert len(set(batch['keys'])) == 1
                    task_report, success, match = self.agent.run(batch, self.evaluator, max_words=self.args.max_words, temp=self.args.temperature)
                    reward = float(success) # + float(match)
                    stats = {'Match': match, 'Success': success}
                    self.agent.update(reward, stats)

                    # supervised learning
                    if self.args.sv_train_freq > 0 and n % self.args.sv_train_freq == 0:
                        self.train_func(self.sys_model, self.sl_train_dataloader, self.args)

                    # record model performance in terms of several evaluation metrics
                    if self.args.record_freq > 0 and n % self.args.record_freq == 0:
                         self.agent.print_dialog(self.agent.dlg_history, reward, stats)
                         print('-'*15, 'Recording start', '-'*15)
                         # save train reward
                         self.learning_exp_file.write('{}\t{}\n'.format(n, np.mean(self.agent.all_rewards[-self.args.record_freq:])))
                         self.learning_exp_file.flush()

                         # PPL & reward on validation
                         valid_loss = self.validate_func(self.sys_model, self.val_dataloader, self.args, use_py=True)
                         v_success, v_match, v_bleu, v_f1 = self.generate_func(self.sys_model, self.val_dataloader, self.args, self.evaluator, None, verbose=False)
                         self.ppl_val_file.write('{}\t{}\t{}\t{}\n'.format(n, np.exp(valid_loss), v_bleu, v_f1))
                         self.ppl_val_file.flush()
                         self.rl_val_file.write('{}\t{}\t{}\t{}\n'.format(n, (v_success + v_match), v_success, v_match))
                         self.rl_val_file.flush()

                         test_loss = self.validate_func(self.sys_model, self.test_dataloader, self.args, use_py=True)
                         t_success, t_match, t_bleu, t_f1 = self.generate_func(self.sys_model, self.test_dataloader, self.args, self.evaluator, None, verbose=False)
                         self.ppl_test_file.write('{}\t{}\t{}\t{}\n'.format(n, np.exp(test_loss), t_bleu, t_f1))
                         self.ppl_test_file.flush()
                         self.rl_test_file.write('{}\t{}\t{}\t{}\n'.format(n, (t_success + t_match), t_success, t_match))
                         self.rl_test_file.flush()

                         # save model is needed
                         if v_success+v_match > best_rewards:
                             print("Model saved with success {} match {}".format(v_success, v_match))
                             torch.save(self.sys_model.state_dict(), self.args.reward_best_model_path)
                             best_rewards = v_success+v_match


                         self.sys_model.train()
                         print('-'*15, 'Recording end', '-'*15)
        except KeyboardInterrupt:
            print("RL training stopped from keyboard")

        print("$$$ Load {}-model".format(self.args.reward_best_model_path))
        self.args.batch_size = 32
        self.sys_model.load_state_dict(torch.load(self.args.reward_best_model_path))

        validate(self.sys_model, self.val_dataloader, self.args, use_py=True)
        validate(self.sys_model, self.test_dataloader, self.args, use_py=True)

        with open(os.path.join(self.args.record_path, 'valid_file.txt'), 'w') as f:
            self.generate_func(self.sys_model, self.val_dataloader, self.args, self.evaluator, num_batch=None, dest_f=f)

        with open(os.path.join(self.args.record_path, 'test_file.txt'), 'w') as f:
            self.generate_func(self.sys_model, self.test_dataloader, self.args, self.evaluator, num_batch=None, dest_f=f)


def validate(model, dataloader, config, batch_cnt=None, use_py=None):
    
    loss_manager = LossManager()
    val_data.epoch_init(config, shuffle=False, verbose=False)
    losses = LossManager()
    while True:
        batch = val_data.next_batch()
        if batch is None:
            break
        if use_py is not None:
            loss = model(batch, mode=TEACH_FORCE, use_py=use_py)
        else:
            loss = model(batch, mode=TEACH_FORCE)

        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

    valid_loss = losses.avg_loss()
    # print('Validation finished at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    self.logger.info(losses.pprint(val_data.name))
    self.logger.info('Total valid loss = {}'.format(valid_loss))
    sys.stdout.flush()
    return valid_loss

def validate_rl(dialog_eval, ctx_gen, num_episode=200):
    print("Validate on training goals for {} episode".format(num_episode))
    reward_list = []
    agree_list = []
    sent_metric = UniquenessSentMetric()
    word_metric = UniquenessWordMetric()
    for _ in range(num_episode):
        ctxs = ctx_gen.sample()
        conv, agree, rewards = dialog_eval.run(ctxs)
        true_reward = rewards[0] if agree else 0
        reward_list.append(true_reward)
        agree_list.append(float(agree if agree is not None else 0.0))
        for turn in conv:
            if turn[0] == 'System':
                sent_metric.record(turn[1])
                word_metric.record(turn[1])
    results = {'sys_rew': np.average(reward_list),
               'avg_agree': np.average(agree_list),
               'sys_sent_unique': sent_metric.value(),
               'sys_unique': word_metric.value()}
    return results


def task_train_single_batch(model, train_data, config):
    batch_cnt = 0
    optimizer = model.get_optimizer(config, verbose=False)
    model.train()

    # decoding CE
    train_data.epoch_init(config, shuffle=True, verbose=False)
    for i in range(16):
        batch = train_data.next_batch()
        if batch is None:
            train_data.epoch_init(config, shuffle=True, verbose=False)
            batch = train_data.next_batch()
        optimizer.zero_grad()
        loss = model(batch, mode=TEACH_FORCE)
        model.backward(loss, batch_cnt)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
