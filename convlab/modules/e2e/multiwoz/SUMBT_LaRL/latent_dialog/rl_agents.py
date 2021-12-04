import torch
import torch.nn as nn
from torch import optim
import numpy as np
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.utils import LONG, FLOAT, Pack
from convlab.modules.dst.multiwoz.dst_util import init_state
import copy

from pytorch_pretrained_bert.optimization import BertAdam
from utils.processor import get_sent, get_bert_sent
from collections import defaultdict
import pdb

EXCLUDED_DOMAIN = ['bus']
EXCLUDED_DOMAIN_SLOT = ['train-book ticket']

class OfflineRlAgent(object):
    def __init__(self, model, corpus, corpus_keys, args, name, tune_pi_only):
        
        self.model = model
        self.corpus = corpus
        self.corpus_keys = corpus_keys
        self.args = args
        self.name = name
        self.raw_goal = None
        self.vec_goals_list = None
        self.logprobs = None
        
        optimizer_grouped_parameters = [p for n, p in self.model.named_parameters() if 'c2z' in n or not tune_pi_only]
        num_train_batches = len(corpus.dataset[0]) # total number of dialogs
        num_train_steps = int(num_train_batches / args.train_batch_size * args.num_train_epochs)
        """
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.rl_lr,
                            warmup=args.warmup_proportion,
                            t_total = num_train_steps)
        """
        self.optimizer = optim.SGD(optimizer_grouped_parameters,
                        lr=self.args.rl_lr,
                        momentum=self.args.momentum,
                        nesterov=(self.args.nesterov and self.args.momentum > 0))
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.0005)
        
        self.num_moving_avg = -1 # no moving average
        self.all_rewards = [] # will be moving averaged
        self.model.train()

    def print_dialog(self, dialog, reward, stats):
        for t_id, turn in enumerate(dialog):
            if t_id % 2 == 0:
                print("Usr: {}".format(' '.join([t for t in turn if t != '<pad>'])))
            else:
                print("Sys: {}".format(' '.join(turn)))
        report = ['{}: {}'.format(k, v) for k, v in stats.items()]
        print("Reward {}. {}".format(reward, report))

    def run(self, batch, evaluator, max_words=None, temp=0.1):
        "TODO: should reformulate"
        self.logprobs = []
        self.dlg_history =[]
        batch_size = len(batch['keys'])
        logprobs, outs = self.model.forward_rl(batch, max_words, temp)
        if batch_size == 1:
            logprobs = [logprobs]
            outs = [outs]

        key = batch['keys'][0]
        sys_turns = []
        # construct the dialog history for printing
        for turn_id, turn in enumerate(batch['contexts']):
            user_input = self.corpus.id2sent(turn[-1])
            self.dlg_history.append(user_input)
            sys_output = self.corpus.id2sent(outs[turn_id])
            self.dlg_history.append(sys_output)
            sys_turns.append(' '.join(sys_output))

        for log_prob in logprobs:
            self.logprobs.extend(log_prob)
        # compute reward here
        generated_dialog = {key: sys_turns}
        return evaluator.evaluateModel(generated_dialog, mode="offline_rl")

    def update(self, reward, stats):
        reward = np.array(reward)
        
        # reward is a list size of batchsize
        if self.num_moving_avg > 0 and len(self.all_rewards) > self.num_moving_avg:
            self.all_rewards.pop(0)
        self.all_rewards.append(reward)
        
        # standardize the reward
        all_rewards = np.concatenate(self.all_rewards)
        r = (reward - np.mean(all_rewards)) / max(1e-4, np.std(all_rewards))
        #r = (reward - np.mean(all_rewards))
        #r = reward
        
        # compute accumulated discounted reward
        rewards = []
        for did in self.logprobs.keys():
            num_turn = len(self.logprobs[did])
            g = r[did]
            r_dial = []
            for _ in range(num_turn):
                r_dial.insert(0, g)
                g = g * self.args.gamma
            rewards.append(r_dial)

        loss = 0
        num_dialog = len(rewards)
        # estimate the loss using one MonteCarlo rollout
        for logprobs, r_dial in zip(self.logprobs.values(), rewards):
            for lp, r in zip(logprobs, r_dial):
                loss -= lp * r

        loss = loss / num_dialog

        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        """
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.policy.c2z.parameters(), self.args.rl_clip)
        self.optimizer.step()
        
        return loss


class OfflineLatentRlAgent(OfflineRlAgent):
    def __init__(self, model, corpus, corpus_keys, args, name, tune_pi_only, bert_tokenizer, dec_vocab):
        super(OfflineLatentRlAgent, self).__init__(model, corpus, corpus_keys, args, name, tune_pi_only)
        self.bert_tokenizer = bert_tokenizer
        self.dec_vocab = dec_vocab
        self.de_tknize = lambda x: ' '.join(x)
        self.belief_state = (init_state())['belief_state']


    def run(self, learning_step, batch, key, evaluator, max_words=None, temp=0.1, use_oracle_bs=False):
        self.logprobs = defaultdict(list)
        self.dlg_history =[]
        
        num_dialog = len(batch[1])
        num_turn = len(batch[1][0])
        batch_size = num_dialog * num_turn
        
        key_idx = batch[0]
        batch = batch[1:]
        
        output = self.model.forward_rl(batch, max_words=max_words)
        if use_oracle_bs:
            state_list = None
        else:
            state_list = self._update_state(output['output_bs'], self.model.ontology, num_dialog, num_turn)
        logprob_z = output['joint_logpz'].view(num_dialog, num_turn, -1)
        pred_labels = output['outs'] # list of num*dialog, num_turn
        if batch_size == 1:
            pred_labels = [pred_labels]
        
        context = batch[0] # user context
        context = context.cpu().numpy()
        
        sys_turns = []
        generated_dialogs = defaultdict(list)
        
        for did in range(num_dialog): 
            for tid in range(num_turn):
                # check padded turn
                if context[did,tid,0] == 0:
                    continue
                
                ctx_str = get_bert_sent(self.bert_tokenizer, context[did, tid, :])
                pred_str = get_sent(self.dec_vocab, self.de_tknize, pred_labels[did * num_turn + tid])

                self.dlg_history.append(ctx_str)
                self.dlg_history.append(pred_str)
                sys_turns.append(pred_str)
                
                if use_oracle_bs:
                    generated_dialogs[key[key_idx[did]]].append((pred_str, None)) # use oracle belief state
                else:
                    generated_dialogs[key[key_idx[did]]].append((pred_str, state_list[did][tid])) # use inferenced belief states

                # append logprob_z of valid turns (i.e., excluding padding turns)
                self.logprobs[did].append(logprob_z[did][tid])

        return evaluator.evaluateModel(generated_dialogs, learning_step, use_oracle_bs=(state_list==None), mode="offline_rl", reduce='none')

    def _update_state(self, output, ontology, num_dialog, num_turn):
        
        # TODO: Cleaning
        new_belief_state_list = []

        for did in range(num_dialog):

            turn_belief_list = []
            for tid in range(num_turn):
        
                new_belief_state = copy.deepcopy(self.belief_state)
                
                # update belief state and Inform action
                for sidx, domain_slot in enumerate(ontology.keys()):

                    domain = domain_slot.split('-')[0].strip()
                    if domain in EXCLUDED_DOMAIN or domain_slot in EXCLUDED_DOMAIN_SLOT:
                        continue

                    slot = slot_normlization(domain_slot.split('-')[1]) # normalize {pricerange, leaveat, arriveby}

                    vidx = torch.argmax(output[sidx][did, tid, :])
                    value = ontology[domain_slot][vidx]

                    if value == 'none':
                        value = ''

                    if ('book' in slot) or (domain_slot =='taxi-departure') or (domain_slot == 'taxi-destination'):
                        # update belief
                        slot = slot.replace('book', '').strip()
                        prev_value = new_belief_state[domain]['book'][slot]
                        if value != prev_value:
                            new_belief_state[domain]['book'][slot] = value

                    else:
                        # update belief
                        prev_value = new_belief_state[domain]['semi'][slot]
                        if value != prev_value:
                            new_belief_state[domain]['semi'][slot] = value

                turn_belief_list.append(new_belief_state)

            new_belief_state_list.append(turn_belief_list)
        
        return new_belief_state_list

def slot_normlization(slot):
    # slots defined self.processor.ontology and self.processor.ontology_request should be matched to slots defined in belief_state (init_belief_state)
    # i.e. arrive by --> 'arriveBy', 'leave at'--> 'leaveAt'

    # Belief state slots
    NORMALIZE_BELIEF_STATE_SLOT = {
        'arrive by': 'arriveBy',
        'leave at': 'leaveAt',
        # 'ticket': '', #note: not matching to init_belief_state
    }
    if slot in NORMALIZE_BELIEF_STATE_SLOT:
        return NORMALIZE_BELIEF_STATE_SLOT[slot]
    else:
        return slot
