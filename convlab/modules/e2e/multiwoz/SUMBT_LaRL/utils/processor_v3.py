## Data processing
## Processing JSON file

import csv
import os
import collections
import json
import logging
from copy import deepcopy
import numpy as np
import torch

from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.corpora import NormMultiWozCorpus
import pdb

PAD = '<pad>'
UNK = '<unk>'
USR = 'YOU:'
SYS = 'THEM:'
BOD = '<d>'
EOD = '</d>'
BOS = '<s>'
EOS = '<eos>'
SEL = '<selection>'
#SPECIAL_TOKENS_DEAL = [PAD, UNK, USR, SYS, BOD, EOS]
#SPECIAL_TOKENS = [PAD, UNK, USR, SYS, BOS, BOD, EOS, EOD]
SPECIAL_TOKENS = [PAD, UNK, EOS, BOS]
STOP_TOKENS = [EOS, SEL]
DECODING_MASKED_TOKENS = [PAD, UNK, USR, SYS, BOD]

DATA_VER = 2

def normalize_slot_name(slot):
    slot = slot.replace('At', ' at').replace('By', ' by')  # Inform slot
    slot = slot.replace('Addr', 'address').replace('Ref', 'reference') # Request slot
    slot = slot.lower()
    return slot

def normalize_inform_act(act):
    # act: domain-act-slot
    slot = act.split('-')[2]
    if slot == 'price':
        slot = 'pricerange'
    elif slot == 'stay':
        slot = 'book stay'
    elif slot == "day":
        slot = 'book day'
    elif slot == "people":
        slot = 'book people'
    elif slot == "time":
        slot = 'book time'
    elif slot == "leave":
        slot = "leave at"
    elif slot == "arrive":
        slot = "arrive by"
    elif slot == "depart":
        slot = "departure"
    elif slot =="dest":
        slot = "destination"
    
    if act == 'train-inform-day':
        slot = 'day'
    
    return 'inform-'+slot
    
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id, output_ids, db_feat):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id
        self.output_ids = output_ids # system
        self.db_feat = db_feat

class Processor(object):
    """Processor for the belief tracking dataset"""
    logger = logging.getLogger()

    def __init__(self, config):
        self.input_template = {'guid':'', 'prev_sys': '', 'user': '', 'sys': '', 'sys_delex':'',
                               'bs': {}, 'user_domain':{}, 'user_act': {},
                               'source':{}, 'kb':'', 'db':{}}
    
        # Ontology for SUMBT
        ontology = json.load(open(os.path.join(config.data_dir, "ontology.json"), "r"))
        for slot in ontology.keys():
            if not "none" in ontology[slot]:
                ontology[slot].append("none")
        ontology_act = json.load(open(os.path.join(config.data_dir, "ontology_act.json"), "r"))
        ontology_request = json.load(open(os.path.join(config.data_dir, "ontology_req.json"), "r"))

        # sorting the ontology according to the alphabetic order of the slots
        self.ontology = collections.OrderedDict(sorted(ontology.items()))
        self.target_slot = list(self.ontology.keys())

        # Belief state
        for slot in self.target_slot:
            self.input_template['bs'][slot] = 'none'

        # domain
        # act (general, inform, request)
        self.ontology_domain = []
        self.ontology_act = []
        self.act_idx=[0] #starting idx of general, inform, request

        for domain_act in ontology_act:
            domain = domain_act.split('-')[0]
            if 'general' in domain:
                self.ontology_act.append(domain_act)
                self.input_template['user_act'][domain_act] = False # initial value is false
            else:
                if domain not in self.ontology_domain:
                    self.ontology_domain.append(domain)
                    self.input_template['user_domain'][domain] = False
        self.act_idx.append(len(self.ontology_act))

        for slot in ontology.keys():
            domain = slot.split('-')[0]
            slot = slot.split('-')[1]
            act = 'inform-'+slot
            if act not in self.ontology_act:
                self.ontology_act.append(act)
                self.input_template['user_act'][act] = False
        self.act_idx.append(len(self.ontology_act))

        for slot in ontology_request:
            domain = slot.split('-')[0]
            slot = slot.split('-')[1]
            act = 'request-'+slot
            if act not in self.ontology_act:
                self.ontology_act.append(act)
                self.input_template['user_act'][act] = False
        
        # Policy and Decoder (For MultiWOZ)
        self.bs_size = len(self.ontology)
        self.db_size = config.db_size
        self.max_vocab_size = config.dec_max_vocab_size
        self.vocab = None
        self.vocab_dict = None
        self.unk_id = None
        self.pad_id = None
        self.dec_max_seq_len = config.dec_max_seq_len
        self.load_dec_dict(config.data_dir)
    
    def load_dec_dict(self, data_dir):
        vocab_dir = os.path.join(data_dir, "dec_vocab.json")
        if os.path.exists(vocab_dir):
            self.vocab_dict = json.load(open(vocab_dir,'r'))
            self.vocab = list(self.vocab_dict)
            self.unk_id = self.vocab_dict[UNK]
            self.pad_id = self.vocab_dict[PAD]

    def get_train_examples(self, data_dir, accumulation=False):
        """See base class."""
        created_dir = os.path.join(data_dir, "train_created.json")
        original_dir =  os.path.join(data_dir, "train.json")

        if os.path.exists(created_dir):
            data = json.load(open(created_dir, 'r'))
            self.load_dec_dict(data_dir)
        else:
            data = self._create_examples(json.load(open(original_dir, 'r')), "train", accumulation)
            json.dump(data, open(created_dir, 'w'), indent=4)
            vocab_dir = os.path.join(data_dir, "dec_vocab.json")
            json.dump(self.vocab_dict, open(vocab_dir, 'w'), indent=4) 

        return data

    def get_dev_examples(self, data_dir, accumulation=False):
        """See base class."""
        created_dir = os.path.join(data_dir, "val_created.json")
        original_dir =  os.path.join(data_dir, "val.json")

        if os.path.exists(created_dir):
            data = json.load(open(created_dir, 'r'))
        else:
            data = self._create_examples(json.load(open(original_dir, 'r')), "dev", accumulation)
            json.dump(data, open(created_dir, 'w'), indent=4)

        return data

    def get_test_examples(self, data_dir, accumulation=False):
        """See base class."""
        created_dir = os.path.join(data_dir, "test_created.json")
        original_dir =  os.path.join(data_dir, "test.json")

        if os.path.exists(created_dir):
            data = json.load(open(created_dir, 'r'))
        else:
            data = self._create_examples(json.load(open(original_dir, 'r')), "test", accumulation)
            json.dump(data, open(created_dir, 'w'), indent=4)

        return data

    def get_labels(self):
        """ return list of value numbers, number of slots (for request), number of general acts """
        return [ self.ontology[slot] for slot in self.target_slot], self.ontology_domain, self.ontology_act

    def _construct_vocab(self, vocab_count):
        # Note: construct vocab using only system utterances
        vocab_count = sorted(vocab_count.items(), key=(lambda x:x[1]), reverse = True)
        raw_vocab_size = len(vocab_count)
        keep_vocab_size = min(self.max_vocab_size, raw_vocab_size)
        
        num_all_words = np.sum([c for t, c in vocab_count]) 
        oov_rate = 100 - 100*np.sum([c for t, c in vocab_count[0:keep_vocab_size]]) / float(num_all_words)

        self.logger.info('cut off at word {} with frequency={},\n'.format(vocab_count[keep_vocab_size - 1][0],
                                                               vocab_count[keep_vocab_size - 1][1]) +
              'OOV rate = {:.2f}%'.format(oov_rate))

        vocab_count = vocab_count[0:keep_vocab_size]
        self.vocab = SPECIAL_TOKENS + [t for t, cnt in vocab_count if t not in SPECIAL_TOKENS]
        self.vocab_dict = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.vocab_dict[UNK]
        self.pad_id = self.vocab_dict[PAD]
        self.logger.info("Raw vocab size {} in train set and final vocab size {}".format(raw_vocab_size, len(self.vocab)))

    def _sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]

    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def _create_examples(self, dialogs, set_type, accumulation=False):
        """Creates examples and consturct decoder vocabulary"""
        examples = []
        req_err_list = {}
        req_num = 0
        req_err = 0
        
        info_err_list = {}
        info_num = 0
        info_err = 0
        
        num_turn = 0
        no_act = 0
        multi_domain = 0
        
        vocab_count = {}
        for did, dialog in dialogs.items():
            prev_sys_text = ''
            tid = 0
            input = deepcopy(self.input_template)
            # processed_goal = self._process_goal(dialog['goal']) #TODO: why goal processing is needed in LaRL?

            for turn in dialog['log']:
                input['guid'] = "%s-%s-%s" % (set_type, did, tid)  # line[0]: dialogue index, line[1]: turn index
                input['prev_sys'] = prev_sys_text

                if len(turn['metadata']) > 0:  # sys turn
                    next_sys_text = turn['text'].lower()
                    next_sys_delex_text = turn['text_delex'].lower()
                    
                    input['sys'] = next_sys_text
                    input['sys_delex'] = next_sys_delex_text
                    
                    # Consstruct decoder vocab
                    if set_type == 'train':
                        for word in next_sys_delex_text.split(): # TODO!!
                            if word in vocab_count:
                                vocab_count[word] += 1
                            else:
                                vocab_count[word] = 1

                    # extract slot-values in belief states
                    for domain, slot_value in turn['metadata'].items():
                        for slot, value in slot_value['semi'].items():
                            slot = normalize_slot_name(slot)

                            if value != '' and value != 'none':
                                domain_slot = domain.lower().strip() + '-' + slot.lower().strip()
                                input['bs'][domain_slot] = value

                        for slot, value in slot_value['book'].items():
                            if slot == 'booked':
                                continue
                            else:
                                slot = normalize_slot_name(slot)
                                domain_slot = domain.lower().strip() + '-book ' + slot.lower().strip()
                                input['bs'][domain_slot] = value
                    
                    # extract db features
                    db_feat_temp = turn['db']
                    new_booking_db_feat = db_feat_temp[-3:]
                    db_feat_temp = db_feat_temp[:-3]
                    for item in new_booking_db_feat:
                        if item == 0:
                            db_feat_temp.extend([1.0, 0.0])
                        else:
                            db_feat_temp.extend([0.0, 1.0])
                    input['db'] = db_feat_temp # Number of found DB search results
                    assert len(db_feat_temp) == 62

                    if 'KB' in turn:
                        input['kb'] = turn['KB']
                        input['source'] = turn['source']
                    else:
                        #print('ERROR: %s' % input['guid'])
                        input['kb'] = 0
                        input['source'] = {}

                    if input['user'] != '':
                        tid += 1
                        examples.append(input)
                        input = deepcopy(self.input_template)
                    prev_sys_text = next_sys_text

                else:  # user turn
                    num_turn += 1
                    input['user'] = turn['text'].lower()
                    
                    if len(turn['dialog_act']) == 0:
                        no_act += 1
                        
                    # extract user dialog act, domain
                    for act, pairs in turn['dialog_act'].items():
                        act = act.lower()
                        
                        domain = act.split('-')[0]
                        if domain in self.ontology_domain:
                            input['user_domain'][domain] = True					
                        
                        if act in self.ontology_act: # general
                            input['user_act'][act] = True
                            
                        elif 'inform' in act:
                            for pair in pairs:
                                domain = act.split('-')[0].lower()
                                slot = normalize_slot_name(pair[0])
                                if slot != 'none':
                                    domain_slot = domain+'-inform-' + slot
                                    domain_slot = normalize_inform_act(domain_slot) # 'inform-slot'
                                    if domain_slot in self.ontology_act: # for some typos
                                        input['user_act'][domain_slot] = True
                                        info_num += 1
                                    else:
                                        if domain_slot not in info_err_list:
                                            info_err_list[domain_slot]=None
                                        info_err += 1
                                        
                        elif 'request' in act:
                            for pair in pairs:
                                domain = act.split('-')[0].lower()
                                slot = normalize_slot_name(pair[0])
                                domain_slot = 'request-' + slot

                                if domain_slot in self.ontology_act: # for some typos
                                    input['user_act'][domain_slot] = True
                                    req_num += 1
                                else:
                                    if domain_slot not in req_err_list:
                                        req_err_list[domain_slot]=None
                                    req_err += 1
                        
                    if sum(input['user_domain'].values()) > 1:
                        multi_domain += 1
                       

        if set_type == 'train':
            self._construct_vocab(vocab_count)

        print("Inform numbers: %d, Errors(Inform slots not exist) : %d" % (info_num, info_err))
        print("Request numbers: %d, Errors(request slots not exist) : %d" % (req_num, req_err))
        print("User Act tagging error: total turns %d: no_act %d (%.3e) multi_domain %d (%.3e) " 
              % (num_turn, no_act, no_act/num_turn, multi_domain, multi_domain/num_turn))
        return examples


    def convert_examples_to_features(self, examples, labels, max_seq_length, tokenizer, max_turn_length):
        """Loads a data file into a list of `InputBatch`s."""

        label_list, domain_list, act_list = labels

        label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
        slot_dim = len(label_list)
        domain_dim = len(domain_list)
        act_dim = len(act_list)
        padding_label = ([-1]*slot_dim, [-1]*domain_dim, [-1]*act_dim)

        features = []
        prev_dialogue_idx = None
        all_padding = [0] * max_seq_length
        all_padding_out = [0] * self.dec_max_seq_len
        all_padding_len = [0, 0]
        all_padding_db = [0] * self.db_size

        max_turn = 0
        for (_, example) in enumerate(examples):
            if max_turn < int(example['guid'].split('-')[2]):
                max_turn = int(example['guid'].split('-')[2])
        max_turn_length = min(max_turn+1, max_turn_length)
        print('max_turn_length %d :' % max_turn_length)

        for (_, example) in enumerate(examples):
            tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example['user'])]
            tokens_b = None
            if example['prev_sys']:
                tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example['prev_sys'])]
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

            try:
                # Belief state label
                label_bs_id = []
                for i, value in enumerate(example['bs'].values()):
                    label_bs_id.append(label_map[i][value])
            except:
                raise ValueError("Invalid slot name %s in domain %s" % (value, i))

            # User domain label
            label_domain_id = []
            for domain in domain_list:
                label_domain_id.append(int(example['user_domain'][domain])) # False=0; True=1

            # User act label
            label_act_id = []
            for act in act_list:
                label_act_id.append(int(example['user_act'][act])) # False=0; True=1

            if DATA_VER == 2:
                # if user act is not annotated, exclude this turn to learn
                if torch.tensor(label_act_id).sum()==0 and torch.tensor(label_domain_id).sum() == 0:
                    label_domain_id = [-1]*domain_dim
                    label_act_id = [-1]*act_dim
            
            label_id = (label_bs_id, label_domain_id, label_act_id)

            curr_dialogue_idx = example['guid'].split('-')[1]
            curr_turn_idx = int(example['guid'].split('-')[2])

            # System delexicalized utterance
            if example['sys_delex']:
                sys_delex = self._sent2id([BOS] + example['sys_delex'].split() + [EOS])
                if len(sys_delex) < self.dec_max_seq_len:
                    sys_delex = sys_delex + [self.pad_id] * (self.dec_max_seq_len - len(sys_delex))
                else:
                    sys_delex = sys_delex[0:self.dec_max_seq_len]
            else:
                sys_delex = [self.pad_id]*self.dec_max_seq_len

            if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
                if prev_turn_idx < max_turn_length:
                    features += [InputFeatures(input_ids=all_padding,
                                            input_len=all_padding_len,
                                            label_id=padding_label,
                                            output_ids = all_padding_out,
                                            db_feat = all_padding_db)
                                ]*(max_turn_length - prev_turn_idx - 1)
                assert len(features) % max_turn_length == 0

            if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
                features.append(InputFeatures(input_ids=input_ids,
                                            input_len=input_len,
                                            label_id=label_id,
                                            output_ids = sys_delex,
                                            db_feat = example['db']))

            prev_dialogue_idx = curr_dialogue_idx
            prev_turn_idx = curr_turn_idx

        if prev_turn_idx < max_turn_length:
            features += [InputFeatures(input_ids=all_padding,
                                    input_len=all_padding_len,
                                    label_id=padding_label,
                                    output_ids = all_padding_out,
                                    db_feat = all_padding_db)]\
                        * (max_turn_length - prev_turn_idx - 1)
        assert len(features) % max_turn_length == 0

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_len= torch.tensor([f.input_len for f in features], dtype=torch.long)
        all_label_bs_ids = torch.tensor([f.label_id[0] for f in features], dtype=torch.long)
        all_label_domain_ids = torch.tensor([f.label_id[1] for f in features], dtype=torch.long)
        all_label_act_ids = torch.tensor([f.label_id[2] for f in features], dtype=torch.long)
        all_output_ids = torch.tensor([f.output_ids for f in features], dtype=torch.long)
        all_db_feat = torch.tensor([f.db_feat for f in features], dtype=torch.long)

        # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
        all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
        all_input_len = all_input_len.view(-1, max_turn_length, 2)
        all_label_bs_ids = all_label_bs_ids.view(-1, max_turn_length, slot_dim)
        all_label_domain_ids = all_label_domain_ids.view(-1, max_turn_length, domain_dim)
        all_label_act_ids = all_label_act_ids.view(-1, max_turn_length, act_dim)
        all_output_ids = all_output_ids.view(-1, max_turn_length, self.dec_max_seq_len)
        all_db_feat = all_db_feat.view(-1, max_turn_length, self.db_size)

        return all_input_ids, all_input_len, [all_label_bs_ids, all_label_domain_ids, all_label_act_ids], all_output_ids, all_db_feat

def get_label_embedding(labels, max_seq_length, tokenizer, device):
    features = []
    for label in labels:
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)

        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)


    return all_label_token_ids, all_label_len

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
def get_sent(vocab, de_tknize, data, stop_eos=True, stop_pad=True):
    ws = []
    for token in data:
        w = vocab[token]
        # TODO EOT
        if (stop_eos and w == EOS) or (stop_pad and w == PAD):
            break
        if w != PAD:
            ws.append(w)

    return de_tknize(ws)

def get_bert_sent(tokenizer, data):
    tokens = tokenizer.convert_ids_to_tokens(data)
    out_string = ' '.join(tokens).replace(' ##', '').strip()
    out_string = out_string.replace('[CLS]', '')
    out_string = out_string.replace('[SEP]', '#')
    out_string = out_string.replace('[PAD]', '')
    
    return out_string