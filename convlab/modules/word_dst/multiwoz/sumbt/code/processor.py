## Data processing
## Processing JSON file

import csv
import os
import collections
import json
from copy import deepcopy
import numpy as np
import torch

import pdb

def normalize_slot_name(slot):
    slot = slot.replace('At', ' at').replace('By', ' by')  # Inform slot
    slot = slot.replace('Addr', 'address').replace('Ref', 'reference') # Request slot
    slot = slot.lower()
    return slot

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id

class Processor(object):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self, config):
        super(Processor, self).__init__()

        self.input_template = {'user': '', 'sys': '', 'bs': {}, 'user_req': {}, 'user_gen': {}, 'guid':''}

        ontology = json.load(open(os.path.join(config.data_dir, "ontology.json"), "r"))
        for slot in ontology.keys():
            if not "none" in ontology[slot]:
                ontology[slot].append("none")
        ontology_act = json.load(open(os.path.join(config.data_dir, "ontology_act.json"), "r"))
        self.ontology_request = json.load(open(os.path.join(config.data_dir, "ontology_req.json"), "r"))

        # sorting the ontology according to the alphabetic order of the slots
        self.ontology = collections.OrderedDict(sorted(ontology.items()))
        self.target_slot = list(self.ontology.keys())

        for slot in self.target_slot:
            self.input_template['bs'][slot] = 'none'

        self.ontology_general = []
        for act in ontology_act:
            if 'general' in act:
                self.ontology_general.append(act)
                self.input_template['user_gen'][act] = False # initial value is false

        for slot in self.ontology_request:
            self.input_template['user_req'][slot] = False

    def get_train_examples(self, data_dir, accumulation=False):
        """See base class."""
        created_dir = os.path.join(data_dir, "train_created.json")
        original_dir =  os.path.join(data_dir, "train.json")

        if os.path.exists(created_dir):
            data = json.load(open(created_dir, 'r'))
        else:
            data = self._create_examples(json.load(open(original_dir, 'r')), "train", accumulation)
            json.dump(data, open(created_dir, 'w'), indent=4)

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
        return [ self.ontology[slot] for slot in self.target_slot], self.ontology_request, self.ontology_general

    def _create_examples(self, dialogs, set_type, accumulation=False):
        """Creates examples for the training and dev sets."""

        examples = []
        req_err = 0
        req_num = 0
        req_err_list = {}
        for did, dialog in dialogs.items():
            prev_sys_text = ''
            tid = 0
            input = deepcopy(self.input_template)

            for turn in dialog['log']:
                input['guid'] = "%s-%s-%s" % (set_type, did, tid)  # line[0]: dialogue index, line[1]: turn index
                input['sys'] = prev_sys_text

                if len(turn['metadata']) > 0:  # sys turn
                    next_sys_text = turn['text'].lower()

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

                    if input['user'] != '':
                        tid += 1
                        examples.append(input)
                        input = deepcopy(self.input_template)
                    prev_sys_text = next_sys_text

                else:  # user turn
                    input['user'] = turn['text'].lower()

                    # extract user dialog act
                    for act, pairs in turn['dialog_act'].items():
                        act = act.lower()

                        if act in self.ontology_general:
                            input['user_gen'][act] = True

                        elif 'request' in act:
                            for pair in pairs:
                                domain = act.split('-')[0].lower()
                                slot = normalize_slot_name(pair[0])
                                domain_slot = domain+ '-' + slot
                                if domain_slot in self.ontology_request: # for some typos
                                    input['user_req'][domain_slot] = True
                                    req_num += 1
                                else:
                                    if domain_slot not in req_err_list:
                                        req_err_list[domain_slot]=None
                                    req_err += 1

        print("Request numbers: %d errors : %d" % (req_num, req_err))
        return examples


def convert_examples_to_features(examples, labels, max_seq_length, tokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    label_list, req_list, gen_list = labels

    label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
    slot_dim = len(label_list)
    req_dim = len(req_list)
    gen_dim = len(gen_list)
    padding_label = ([-1]*slot_dim, [-1]*req_dim, [-1]*gen_dim)

    features = []
    prev_dialogue_idx = None
    all_padding = [0] * max_seq_length
    all_padding_len = [0, 0]

    max_turn = 0
    for (ex_index, example) in enumerate(examples):
        if max_turn < int(example['guid'].split('-')[2]):
            max_turn = int(example['guid'].split('-')[2])
    max_turn_length = min(max_turn+1, max_turn_length)
    print('max_turn_length %d :' % max_turn_length)

    for (ex_index, example) in enumerate(examples):
        tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example['user'])]
        tokens_b = None
        if example['sys']:
            tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example['sys'])]
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
            pdb.set_trace()

        # User request label
        label_req_id = []
        for slot in req_list:
            label_req_id.append(int(example['user_req'][slot])) # False=0; True=1

        # User act (general) label
        label_act_id = []
        for slot in gen_list:
            label_act_id.append(int(example['user_gen'][slot])) # False=0; True=1

        label_id = (label_bs_id, label_req_id, label_act_id)

        curr_dialogue_idx = example['guid'].split('-')[1]
        curr_turn_idx = int(example['guid'].split('-')[2])

        if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
            if prev_turn_idx < max_turn_length:
                features += [InputFeatures(input_ids=all_padding,
                                           input_len=all_padding_len,
                                           label_id=padding_label)
                             ]*(max_turn_length - prev_turn_idx - 1)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_len=input_len,
                              label_id=label_id))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

    if prev_turn_idx < max_turn_length:
        features += [InputFeatures(input_ids=all_padding,
                                   input_len=all_padding_len,
                                   label_id=padding_label)]\
                    * (max_turn_length - prev_turn_idx - 1)
    assert len(features) % max_turn_length == 0

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_len= torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_bs_ids = torch.tensor([f.label_id[0] for f in features], dtype=torch.long)
    all_label_req_ids = torch.tensor([f.label_id[1] for f in features], dtype=torch.long)
    all_label_gen_ids = torch.tensor([f.label_id[2] for f in features], dtype=torch.long)

    # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)
    all_label_bs_ids = all_label_bs_ids.view(-1, max_turn_length, slot_dim)
    all_label_req_ids = all_label_req_ids.view(-1, max_turn_length, req_dim)
    all_label_gen_ids = all_label_gen_ids.view(-1, max_turn_length, gen_dim)

    return all_input_ids, all_input_len, [all_label_bs_ids, all_label_req_ids, all_label_gen_ids]


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
