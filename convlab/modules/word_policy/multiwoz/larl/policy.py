#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
LaRL word policy.

Glossary
--------
usr: user
sys: system
utt: utterance
delex: delexicalize / delexicalized
"""
from __future__ import division, print_function, unicode_literals

import os
import pickle
import re
import shutil
import tempfile
import zipfile
from copy import deepcopy
import json
import random

import numpy as np

import torch

from convlab.lib import logger
from convlab.lib.file_util import cached_path
from convlab.modules.dst.multiwoz.dst_util import init_state
from convlab.modules.policy.system.policy import SysPolicy
from convlab.modules.word_policy.multiwoz.mdrg.utils import util
from convlab.modules.word_policy.multiwoz.mdrg.utils.nlp import normalize
from convlab.modules.word_policy.multiwoz.mdrg.policy import delexicaliseReferenceNumber
from convlab.modules.word_policy.multiwoz.mdrg.policy import get_summary_bstate

from convlab.modules.word_policy.multiwoz.larl.latent_dialog.utils import (
    Pack,
    set_seed,
)
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.corpora import (
    NormMultiWozCorpus,
)
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.models_task import (
    SysPerfectBD2Cat,
)
# from convlab.modules.word_policy.multiwoz.larl.latent_dialog.normalizer import delexicalize
from convlab.modules.word_policy.multiwoz.mdrg.utils import delexicalize
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.corpora import EOS, PAD

from nltk.stem.porter import *

stemmer = PorterStemmer()

# TODO(seungjaeryanlee): Make it configurable?
USE_GPU = False


domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
dbs = {}
for domain in domains:
    dbs[domain] = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        '../../../../data/multiwoz/db/{}_db.json'.format(domain))))

def query(domain, constraints, ignore_open=True):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    if domain == 'taxi':
        return [{'taxi_colors': random.choice(dbs[domain]['taxi_colors']), 
        'taxi_types': random.choice(dbs[domain]['taxi_types']), 
        'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
    if domain == 'police':
        return dbs['police']
    #if domain == 'hospital':
    #    return dbs['hospital']

    found = []
    for i, record in enumerate(dbs[domain]):
        for key, val in constraints:
            if type(val) == list:
                val_temp0 = val[0]
                val_temp1 = val[1]
                try:
                    record_keys = [key.lower() for key in record]
                    if key.lower() not in record_keys and stemmer.stem(key) not in record_keys:
                        continue
                    if key == 'leaveAt':
                        val1_0 = int(val_temp0.split(':')[0]) * 100 + int(val_temp0.split(':')[1])
                        val1_1 = int(val_temp1.split(':')[0]) * 100 + int(val_temp1.split(':')[1])
                        val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                        if val1_0 > val2 and val1_1 > val2:
                            break
                    elif key == 'arriveBy':
                        val1_0 = int(val_temp0.split(':')[0]) * 100 + int(val_temp0.split(':')[1])
                        val1_1 = int(val_temp1.split(':')[0]) * 100 + int(val_temp1.split(':')[1])
                        val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                        if val1_0 < val2 and val1_1 < val2:
                            break
                    # elif ignore_open and key in ['destination', 'departure', 'name']:
                    elif ignore_open and key in ['destination', 'departure']:
                        continue
                    else:
                        if val_temp0.strip() != record[key].strip() and val_temp1.strip() != record[key].strip():
                            break
                except:
                    continue
            else:
                if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                    pass
                else:
                    try:
                        record_keys = [key.lower() for key in record]
                        if key.lower() not in record_keys and stemmer.stem(key) not in record_keys:
                            continue
                        if key == 'leaveAt':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                            if val1 > val2:
                                break
                        elif key == 'arriveBy':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                            if val1 < val2:
                                break
                        # elif ignore_open and key in ['destination', 'departure', 'name']:
                        elif ignore_open and key in ['destination', 'departure']:
                            continue
                        else:
                            if val.strip() != record[key].strip():
                                break
                    except:
                        continue
        else:
            record['reference'] = f'{i:08d}'
            found.append(record)

    return found
    '''
    if len(found) == len(dbs[domain]):
        return []
    else:
        return found
    '''


def addBookingPointer(state, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if "book" in state['restaurant']:
        if "booked" in state['restaurant']['book']:
            if state['restaurant']['book']["booked"]:
                if "reference" in state['restaurant']['book']["booked"][0]:
                    rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if "book" in state['hotel']:
        if "booked" in state['hotel']['book']:
            if state['hotel']['book']["booked"]:
                if "reference" in state['hotel']['book']["booked"][0]:
                    hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if "book" in state['train']:
        if "booked" in  state['train']['book']:
            if state['train']['book']["booked"]:
                if "reference" in state['train']['book']["booked"][0]:
                    train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    # pprint(pointer_vector)
    return pointer_vector


def addDBPointer(state):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    db_results = {}
    num_entities = {} 
    for domain in domains:
        # entities = dbPointer.queryResultVenues(domain, {'metadata': state})
        if domain != 'train':
            entities = query(domain, state[domain]['semi'].items())
        else:
            entities = query(domain, state[domain]['semi'].items(), ignore_open=False)
        num_entities[domain] = len(entities)
        if len(entities) > 0: 
            # fields = dbPointer.table_schema(domain)
            # db_results[domain] = dict(zip(fields, entities[0]))
            db_results[domain] = entities[0]
        # pointer_vector = dbPointer.oneHotVector(len(entities), domain, pointer_vector)
        pointer_vector = oneHotVector(len(entities), domain, pointer_vector)

    return pointer_vector, db_results, num_entities 

def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0,0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector


def populate_template(template, top_results, num_results, state):
    active_domain = None if len(top_results.keys()) == 0 else list(top_results.keys())[0]
    #template = template.replace('book [value_count] of them', 'book one of them')
    template = template.replace('book [value_count] of', 'book one of')
    template = template.replace('book [value_count] for you', 'book one for you')
    tokens = template.split()
    response = []
    for token in tokens:
        if token.startswith('[') and token.endswith(']'):
            domain = token[1:-1].split('_')[0]
            slot = token[1:-1].split('_')[1]
            if domain == 'train' and slot == 'id':
                slot = 'trainID'
            if domain in top_results and len(top_results[domain]) > 0 and slot in top_results[domain]:
                if domain != 'taxi':
                    # print('{} -> {}'.format(token, top_results[domain][slot]))
                    response.append(top_results[domain][slot])
                else:
                    if slot == 'type':
                        response.append(" ".join(top_results['taxi_colors'], top_results['taxi_types']))
                    else:
                        response.append(top_results[domain][slot])
            elif domain == 'value':
                if slot == 'count':
                    response.append(str(num_results))
                elif slot == 'place':
                    if 'arrive' in response:
                        for d in state: 
                            if d == 'history':
                                continue
                            if 'destination' in state[d]['semi']:
                                response.append(state[d]['semi']['destination'])
                                break
                    elif 'leave' in response:
                        for d in state: 
                            if d == 'history':
                                continue
                            if 'departure' in state[d]['semi']:
                                response.append(state[d]['semi']['departure'])
                                break
                    else:
                        try:
                            for d in state: 
                                if d == 'history':
                                    continue
                                for s in ['destination', 'departure']:
                                    if s in state[d]['semi']:
                                        response.append(state[d]['semi'][s])
                                        raise
                        except:
                            pass
                        else:
                            response.append(token)
                elif slot == 'time':
                    if 'arrive' in ' '.join(response[-3:]):
                        if active_domain is not None and 'arriveBy' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                            response.append(top_results[active_domain]['arriveBy'])
                            continue 
                        for d in state: 
                            if d == 'history':
                                continue
                            if 'arriveBy' in state[d]['semi']:
                                response.append(state[d]['semi']['arriveBy'])
                                break
                    elif 'leave' in ' '.join(response[-3:]):
                        if active_domain is not None and 'leaveAt' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                            response.append(top_results[active_domain]['leaveAt'])
                            continue 
                        for d in state: 
                            if d == 'history':
                                continue
                            if 'leaveAt' in state[d]['semi']:
                                response.append(state[d]['semi']['leaveAt'])
                                break
                    elif 'book' in response:
                        if state['restaurant']['book']['time'] != "":
                            response.append(state['restaurant']['book']['time'])
                    else:
                        try:
                            for d in state: 
                                if d == 'history':
                                    continue
                                for s in ['arriveBy', 'leaveAt']:
                                    if s in state[d]['semi']:
                                        response.append(state[d]['semi'][s])
                                        raise
                        except:
                            pass
                        else:
                            response.append(token)
                else:
                    # slot-filling based on query results 
                    for d in top_results:
                        if slot in top_results[d]:
                            response.append(top_results[d][slot])
                            break
                    else:
                        # slot-filling based on belief state
                        for d in state:
                            if d == 'history':
                                continue
                            if slot in state[d]['semi']:
                                response.append(state[d]['semi'][slot])
                                break
                        else:
                            response.append(token)
            else:
                # print(token)
                response.append(token)
        else:
            response.append(token)

    try:
        response = ' '.join(response)
    except Exception as e:
        print(e)
        import pprint
        pprint.pprint(response)
        raise
    response = response.replace(' -s', 's')
    response = response.replace(' -ly', 'ly')
    response = response.replace(' .', '.')
    response = response.replace(' ?', '?')
    return response


def mark_not_mentioned(state):
    for domain in state:
        # if domain == 'history':
        if domain not in ['police', 'hospital', 'taxi', 'train', 'attraction', 'restaurant', 'hotel']:
            continue
        try:
            # if len([s for s in state[domain]['semi'] if s != 'book' and state[domain]['semi'][s] != '']) > 0:
                # for s in state[domain]['semi']:
                #     if s != 'book' and state[domain]['semi'][s] == '':
                #         state[domain]['semi'][s] = 'not mentioned'
            for s in state[domain]['semi']:
                if state[domain]['semi'][s] == '':
                    state[domain]['semi'][s] = 'not mentioned'
        except Exception as e:
            # print(str(e))
            # pprint(state[domain])
            pass


def get_active_domain(prev_active_domain, prev_state, state):
    domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'hospital', 'police']
    active_domain = None
    # print('get_active_domain')
    for domain in domains:
        '''
        if domain not in prev_state and domain not in state:
            continue
        if domain in prev_state and domain not in state:
            return domain
        elif domain not in prev_state and domain in state:
            return domain
        elif prev_state[domain] != state[domain]:
            active_domain = domain
        '''
        if prev_state[domain] != state[domain]:
            active_domain = domain

    if active_domain is None:
        if prev_active_domain is not None:
            active_domain = prev_active_domain
    return active_domain 


class LaRLWordPolicy(SysPolicy):
    def __init__(self, pretrained_folder, model_path):
        """Latent Reinforcement Learning word policy.

        Rethinking Action Spaces for Reinforcement Learning in End-to-end Dialog Agents with Latent Variable Models
        https://arxiv.org/abs/1902.08858

        Parameters
        ----------
        pretrained_folder : str
            The folder inside "sys_config_log_model" folder that contains the pretrained model.
            The "sys_config_log_model" should be located in repository root (where run.py exists).
                For example: "2019-08-20-04-17-55-sl_cat"

        model_path : str
            The relative model path from repo root.
                For example: "sys_config_log_model/2019-08-20-04-17-55-sl_cat/rl-2019-08-23-04-06-55/reward_best.model"

        """
        # NOTE(seungjaeryanlee): Extracted MDRG's dic
        with open("data/larl/dic.json") as f:
            self.dic = json.load(f)

        # NOTE(seungjaeryanlee): Custom extracted JSON files from original LaRL corpora
        # TODO(seungjaeryanlee): Check if LaRL has different dict for input and output
        with open("data/larl/index2word.json") as f:
            self.index2word = json.load(f)
        with open("data/larl/word2index.json") as f:
            self.word2index = json.load(f)
        with open("data/larl/vocab.json") as f:
            self.vocab = json.load(f)

        # Load LaRL model
        self.larl_model = self._load_model(pretrained_folder, model_path)

        self.prev_state = init_state()
        self.prev_active_domain = None

    def init_session(self):
        self.prev_state = init_state()
        self.prev_active_domain = None

    def _load_model(self, pretrained_folder, model_path):
        """Load pretrained model for evaluation."""

        # Load training configuration
        train_config = Pack(json.load(open(os.path.join("sys_config_log_model", pretrained_folder, "config.json"))))
        train_config["dropout"] = 0.0
        train_config["use_gpu"] = USE_GPU

        # Create model with fake corpus
        mock_corpus = Pack(vocab=self.vocab, vocab_dict=self.word2index, bs_size=94, db_size=30)
        model = SysPerfectBD2Cat(mock_corpus, train_config)

        # Load model
        model.load_state_dict(torch.load(model_path))

        return model

    def predict(self, state):
        """Generate next response.

        Parameters
        ----------
        state : dict
            State after passing NLU and DST.

                ```
                {
                    "user_action": { ... },
                    "belief_state": { ... },
                    "request_state": { ... },
                    "history": [ ... , [ sys_utt, usr_utt ], ... ],
                }
                ```

            The state["history"][0][0] is null.

        Returns
        -------
        response : str
            Response generated by LaRL. The slots are not filled/

                ```
                the phone number is [hotel_phone].
                ```

        """

        # ██████╗ ███████╗    ███╗   ██╗    ██████╗ ██████╗ 
        # ██╔══██╗██╔════╝    ████╗  ██║    ██╔══██╗██╔══██╗
        # ██████╔╝███████╗    ██╔██╗ ██║    ██║  ██║██████╔╝
        # ██╔══██╗╚════██║    ██║╚██╗██║    ██║  ██║██╔══██╗
        # ██████╔╝███████║    ██║ ╚████║    ██████╔╝██████╔╝
        # ╚═════╝ ╚══════╝    ╚═╝  ╚═══╝    ╚═════╝ ╚═════╝ 
        # Vectorize belief state (BS) and database results (DB)

        prev_belief_state = deepcopy(self.prev_state["belief_state"])
        belief_state = deepcopy(state["belief_state"])

        # Belief state semi: '' -> 'not mentioned'
        mark_not_mentioned(prev_belief_state)
        mark_not_mentioned(belief_state)

        # Add database pointer
        pointer_vector, top_results_temp, num_results_temp = addDBPointer(belief_state)
        # Add booking pointer
        pointer_vector = addBookingPointer(belief_state, pointer_vector)
        belief_summary = get_summary_bstate(belief_state)

        bs_tensor = np.array([belief_summary])
        db_tensor = np.array([pointer_vector])

        #  ██████╗ ██████╗ ███╗   ██╗████████╗███████╗██╗  ██╗████████╗
        # ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
        # ██║     ██║   ██║██╔██╗ ██║   ██║   █████╗   ╚███╔╝    ██║   
        # ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══╝   ██╔██╗    ██║   
        # ╚██████╗╚██████╔╝██║ ╚████║   ██║   ███████╗██╔╝ ██╗   ██║   
        #  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   
        # Create context vector from last user and system utterances.

        # Last utterances
        sys, usr = state["history"][-1]

        sys = normalize(sys)
        usr = normalize(usr)

        # Delexicalize
        usr = delexicalize.delexicalise(" ".join(usr.split()), self.dic)
        sys = delexicalize.delexicalise(" ".join(sys.split()), self.dic)

        # parsing reference number GIVEN belief state
        usr = delexicaliseReferenceNumber(usr, belief_state)
        sys = delexicaliseReferenceNumber(sys, belief_state)

        # Replace numbers with [value_count]
        digitpat = re.compile("\d+")
        usr = re.sub(digitpat, "[value_count]", usr)
        sys = re.sub(digitpat, "[value_count]", sys)

        logger.delex(f"Delexicalized user utterance: {usr}")

        # TODO(seungjaeryanlee): Check this processing!!!
        UNK_id = 1
        sys = [
            self.word2index[word] if word in self.word2index else UNK_id
            for word in normalize(sys).strip(" ").split(" ")
        ] + [util.EOS_token]
        usr = [
            self.word2index[word] if word in self.word2index else UNK_id
            for word in normalize(usr).strip(" ").split(" ")
        ] + [util.EOS_token]
        input_tensor = [sys, usr]
        input_tensor, input_lengths = util.padSequence(input_tensor)


        # ███████╗ ██████╗ ██████╗ ██╗    ██╗ █████╗ ██████╗ ██████╗ 
        # ██╔════╝██╔═══██╗██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔══██╗
        # █████╗  ██║   ██║██████╔╝██║ █╗ ██║███████║██████╔╝██║  ██║
        # ██╔══╝  ██║   ██║██╔══██╗██║███╗██║██╔══██║██╔══██╗██║  ██║
        # ██║     ╚██████╔╝██║  ██║╚███╔███╔╝██║  ██║██║  ██║██████╔╝
        # ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 

        data_feed = {
            "contexts": np.expand_dims(input_tensor, 0),
            "context_lens": torch.LongTensor(np.array([2])),
            # NOTE(seungjaeryanlee): data_feeds["outputs"] is never used when mode is "gen"
            "outputs": np.zeros((1, 1)),
            "bs": bs_tensor,
            "db": db_tensor,
        }

        # TODO(seungjaeryanlee): Validate forward_eval()
        ret_dict, _ = self.larl_model.forward(data_feed, mode='gen', clf=False, gen_type='greedy', use_py=None, return_latent=False)
        output_words = ' '.join([self.index2word[str(t.item())] for t in ret_dict['sequence']])

        # NOTE(seungjaeryanlee): Cut sentence after EOS or PAD like LaRL (latent_dialog/main.py get_sent() L532)
        if EOS in output_words:
            output_words = output_words[0:output_words.index(EOS)]
        if PAD in output_words:
            output_words = output_words[0:output_words.index(PAD)]

        active_domain = get_active_domain(
            self.prev_active_domain, prev_belief_state, belief_state
        )
        
        #print('==================active domain {}==================prev active domain {}=================='.format(active_domain, self.prev_active_domain))
        domains_ex = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
        top_results = {}
        num_results = {} 
        for domain in domains_ex:
            if domain != 'train':
                entities = query(domain, belief_state[domain]['semi'].items())
            else:
                entities = query(domain, belief_state[domain]['semi'].items(), ignore_open=False)
            num_results[domain] = len(entities)
            if len(entities) > 0:
                top_results[domain] = entities[0]


        if active_domain is not None and active_domain in num_results:
            num_results = num_results[active_domain]
        else:
            num_results = 0
        
        if active_domain is not None and active_domain in top_results:
            top_results = {active_domain: top_results[active_domain]}
        else:
            top_results = {}
        

        logger.db(f"Top DB results: {top_results}")
        logger.db(f"Number of DB results: {num_results}")

        logger.delex(f"Delexicalized system utterance: {output_words}")

        # TODO(seungjaeryanlee): Remove this after fixing error
        # File "/mnt/rlee0201/git/ConvLab/convlab/modules/word_policy/multiwoz/mdrg/policy.py", line 397, in populate_template
        #     response = ' '.join(response)
        # TypeError: sequence item 11: expected str instance, dict found
        try:
            response = populate_template(output_words, top_results, num_results, belief_state)
        except:
            response = "Sorry. Can you repeat that?"

        self.prev_state = deepcopy(state)
        self.prev_active_domain = active_domain

        return response
