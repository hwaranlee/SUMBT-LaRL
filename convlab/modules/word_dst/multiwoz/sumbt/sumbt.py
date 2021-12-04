# Copyright 2019 SKT/T-BRAIN/Conv.AI Prj
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import logging
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import convlab
from convlab.modules.dst.multiwoz.dst_util import init_state, init_belief_state
from convlab.modules.dst.state_tracker import Tracker
from convlab.modules.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA

from pytorch_pretrained_bert.tokenization import BertTokenizer
from .code.sumbt_unified import BeliefTracker as SUMBT
from .code.processor import truncate_seq_pair

EXCLUDED_DOMAIN = ['bus']
EXCLUDED_DOMAIN_SLOT = ['train-book ticket']

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SUMBTTracker(Tracker):

    def __init__(self, model_dir="models/sumbt"):
        Tracker.__init__(self)

        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(self.device, self.n_gpu))

        config = json.load(open(os.path.join(self.model_dir, 'config.json')))
        self.config = Namespace(**config)

        self.ontology = self.config.ontology
        self.ontology_request = self.config.labels[1]
        self.ontology_general = self.config.labels[2]

        vocab_dir = os.path.join(self.config.bert_dir, '%s-vocab.txt' % self.config.bert_model)
        if not os.path.exists(vocab_dir):
            raise ValueError("Can't find %s " % vocab_dir)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case = self.config.do_lower_case)

        self.model = SUMBT(args=self.config, labels=None, device=self.device, model_dir=model_dir)
        self.load_model()
        self.model.eval()
        self.max_seq_length = self.config.max_seq_length
        self.threshold = 0.5

        self.state = init_state()
        prefix = os.path.dirname(os.path.dirname(convlab.__file__))
        self.value_dict = json.load(open(prefix+'/data/multiwoz/value_dict.json'))

        self.context_input_ids = None
        self.context_input_len = None

    def load_model(self):
        model_file = os.path.join(self.model_dir, 'pytorch_model.bin')
        if not os.path.isfile(model_file):
            raise Exception("No model for SUMBT is specified!")

        logger.info('Load SUMBT from %s' % model_file)

        # in the case that slot and values are different between the training and evaluation
        ptr_model = torch.load(model_file)

        if self.n_gpu == 1:
            state = self.model.state_dict()
            state.update(ptr_model)
            self.model.load_state_dict(state)
        else:
            raise Exception("Multi-gpu setting is not provided now")
            self.model.module.load_state_dict(ptr_model)

        self.model.to(self.device)

    def update(self, user_utter=None):
        """
        Update dialog state based on new user dialog act.
        Args:
            user_act (str): The dialog act (or utterance) of user input. The class of user_act depends on
                    the method of state tracker. For example, for rule-based tracker, type(user_act) == dict; while for
                    MDBT, type(user_act) == str.
        Returns:
            new_state (dict): Updated dialog state, with the same form of previous state. Note that the dialog state is
                    also a private data member.
        """

        if not isinstance(user_utter, str):
            raise Exception('Expected user_act to be <class \'str\'> type, but get {}.'.format(type(user_utter)))

        turn_level = False

        if turn_level:
            return self._update_turn_level(user_utter)
        else:
            return self._update_context_level(user_utter)

    def _update_context_level(self, user_utter=None):

        prev_state = self.state
        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        new_request_state = copy.deepcopy(prev_state['request_state'])
        user_action = {}

        # preprocessing
        sys_utter = self.state['history'][-1][0]
        if sys_utter == 'null':
            sys_utter = None

        input_ids, input_len = turn_preprocess(user_utter, sys_utter,
                                               max_seq_length = self.max_seq_length,
                                               max_turn_length = 1,
                                               tokenizer = self.tokenizer)

        input_ids = input_ids.to(self.device)
        input_len = input_len.to(self.device)

        if len(self.state['history']) == 1:
            self.context_input_ids = input_ids
            self.context_input_len = input_len
        else:
            self.context_input_ids = torch.cat([self.context_input_ids, input_ids], dim=1)
            self.context_input_len = torch.cat([self.context_input_len, input_len], dim=1)
            input_ids = self.context_input_ids
            input_len = self.context_input_len

        # forward model
        output, out_req, out_gen = self.model(input_ids, input_len)
        out_req = torch.sigmoid(out_req[0, -1,:]).squeeze()
        out_gen = torch.sigmoid(out_gen[0, -1,:]).squeeze()

        # update belief state and Inform action
        for sidx, domain_slot in enumerate(self.ontology.keys()):

            domain = domain_slot.split('-')[0].strip()
            if domain in EXCLUDED_DOMAIN or domain_slot in EXCLUDED_DOMAIN_SLOT:
                continue

            slot = slot_normlization(domain_slot.split('-')[1]) # normalize {pricerange, leaveat, arriveby}

            vidx = torch.argmax(output[sidx][0,-1,:])
            value = self.ontology[domain_slot][vidx]

            if value == 'none':
                value = ''

            if ('book' in slot) or (domain_slot =='taxi-departure') or (domain_slot == 'taxi-destination'):
                # update belief
                slot = slot.replace('book', '').strip()
                prev_value = new_belief_state[domain]['book'][slot]
                if value != prev_value:
                    new_belief_state[domain]['book'][slot] = value

                    # update action
                    domain, slot = act_domain_slot_normalization(domain, slot)
                    if (domain + "-Inform") not in user_action:
                        user_action[domain + "-Inform"] = []
                    if [slot, value] not in user_action[domain + "-Inform"]:
                        user_action[domain + "-Inform"].append([slot, value])

            else:
                # update belief
                # value = normalize_value(self.value_dict, domain.lower(), slot.lower(), value)
                prev_value = new_belief_state[domain]['semi'][slot]
                if value != prev_value:
                    new_belief_state[domain]['semi'][slot] = value

                    # update action
                    domain, slot = act_domain_slot_normalization(domain, slot)
                    if (domain + "-Inform") not in user_action:
                        user_action[domain + "-Inform"] = []
                    if [slot, value] not in user_action[domain + "-Inform"]:
                        user_action[domain + "-Inform"].append([slot, value])

        # update request state and Request action
        for sidx, domain_slot in enumerate(self.ontology_request):
            if out_req[sidx] > self.threshold:
                # domain_slot preprocessing
                domain = domain_slot.split('-')[0]
                slot = slot_normlization(domain_slot.split('-')[1])  # normalize {pricerange, leaveat, arriveby}

                # update action
                domain, slot = act_domain_slot_normalization(domain, slot)
                if (domain + "-Request") not in user_action:
                    user_action[domain + "-Request"] = []
                if [slot, '?'] not in user_action[domain + "-Request"]:
                    user_action[domain + "-Request"].append([slot, '?'])

                # update request state
                slot = slot.lower()
                domain  = domain.lower()
                if not domain in new_request_state:
                    new_request_state[domain] = {}
                if slot not in new_request_state[domain]:
                    new_request_state[domain][slot] = 0

        # update state
        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state

        # update user general act
        for sidx, domain_slot in enumerate(self.ontology_general):
            # find value
            if out_gen[sidx] > self.threshold:
                if not domain_slot in user_action:
                    user_action[domain_slot] = []
                user_action[domain_slot].append(["none", "none"])

        # update user act
        new_state['user_action'] = user_action

        self.state = new_state
        return self.state

    def _update_turn_level(self, user_utter=None):

        prev_state = self.state
        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        new_request_state = copy.deepcopy(prev_state['request_state'])
        user_action = {}

        # preprocessing
        sys_utter = self.state['history'][-1][0]
        if sys_utter == 'null':
            sys_utter = None

        input_ids, input_len = turn_preprocess(user_utter, sys_utter,
                                               max_seq_length = self.max_seq_length,
                                               max_turn_length = 1,
                                               tokenizer = self.tokenizer)
        input_ids = input_ids.to(self.device)
        input_len = input_len.to(self.device)

        # forward model (Note: 1 dialog, 1 turn is assumed)
        output, out_req, out_gen = self.model(input_ids, input_len)

        out_req = torch.sigmoid(out_req).squeeze()
        out_gen = torch.sigmoid(out_gen).squeeze()

        # update belief state and Inform action
        for sidx, domain_slot in enumerate(self.ontology.keys()):

            domain = domain_slot.split('-')[0].strip()
            if domain in EXCLUDED_DOMAIN or domain_slot in EXCLUDED_DOMAIN_SLOT:
                continue

            slot = slot_normlization(domain_slot.split('-')[1]) # normalize {pricerange, leaveat, arriveby}

            vidx = torch.argmax(output[sidx])
            value = self.ontology[domain_slot][vidx]

            if value != 'none':
                if ('book' in slot) or (domain_slot =='taxi-departure') or (domain_slot == 'taxi-destination'):
                    # update belief
                    slot = slot.replace('book', '').strip()
                    new_belief_state[domain]['book'][slot] = value

                    # update action
                    domain, slot = act_domain_slot_normalization(domain, slot)
                    if (domain + "-Inform") not in user_action:
                        user_action[domain + "-Inform"] = []
                    if [slot, value] not in user_action[domain + "-Inform"]:
                        user_action[domain + "-Inform"].append([slot, value])

                else:
                    # update belief
                    new_belief_state[domain.lower()]['semi'][slot] = value

                    # update action
                    domain, slot = act_domain_slot_normalization(domain, slot)
                    if (domain + "-Inform") not in user_action:
                        user_action[domain + "-Inform"] = []
                    if [slot, value] not in user_action[domain + "-Inform"]:
                        user_action[domain + "-Inform"].append([slot, value])

        # update request state and Request action
        for sidx, domain_slot in enumerate(self.ontology_request):
            if out_req[sidx] > self.threshold:
                # domain_slot preprocessing
                domain = domain_slot.split('-')[0]
                slot = slot_normlization(domain_slot.split('-')[1])  # normalize {pricerange, leaveat, arriveby}

                # update action
                domain, slot = act_domain_slot_normalization(domain, slot)
                if (domain + "-Request") not in user_action:
                    user_action[domain + "-Request"] = []
                if [slot, '?'] not in user_action[domain + "-Request"]:
                    user_action[domain + "-Request"].append([slot, '?'])

                # update request state
                slot = slot.lower()
                domain  = domain.lower()
                if not domain in new_request_state:
                    new_request_state[domain] = {}
                if slot not in new_request_state[domain]:
                    new_request_state[domain][slot] = 0

        # update state
        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state

        # update user general act
        for sidx, domain_slot in enumerate(self.ontology_general):
            # find value
            if out_gen[sidx] > 0.5:
                if not domain_slot in user_action:
                    user_action[domain_slot] = []
                user_action[domain_slot].append(["none", "none"])

        # update user act
        new_state['user_action'] = user_action

        self.state = new_state
        return self.state

    def init_session(self):
        """Init the Tracker to start a new session."""
        self.state = init_state()
        self.context_input_ids = None
        self.context_input_len = None

def slot_normlization(slot):
    # slots defined self.ontology and self.ontology_request should be matched to slots defined in belief_state (init_belief_state)
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

def act_domain_slot_normalization(domain, slot):
    # Act slots
    # slots defined self.ontology and self.ontology_request should be matched to values of REF_USR_DA[domain]
    # i.e. address --> 'Addr', reference --> 'Ref'
    
    NORMALIZE_ACT_SLOT = {
        'address':'addr',
        'reference': 'ref',
        'pricerange': 'price',
        'leaveAt': 'leave',
        'arriveBy': 'arrive',
        'destination': 'dest',
        'departure': 'depart',
        'pricerange': 'price'
    }

    if slot in NORMALIZE_ACT_SLOT:
        slot = NORMALIZE_ACT_SLOT[slot]

    return domain.capitalize(), slot.capitalize()

def turn_preprocess(user_utter, sys_utter, max_seq_length, max_turn_length, tokenizer):

    user_utter = user_utter.lower()
    tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(user_utter)]
    tokens_b = None
    if sys_utter:
        sys_utter = sys_utter.lower()
        tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(sys_utter)]
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    input_len = [len(tokens), 0]

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        input_len[1] = len(tokens_b) + 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    all_input_ids = torch.tensor([input_ids], dtype=torch.long)
    all_input_len = torch.tensor([input_len], dtype=torch.long)

    # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)

    return all_input_ids, all_input_len
