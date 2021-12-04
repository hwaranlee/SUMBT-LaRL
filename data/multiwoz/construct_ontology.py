# This script convert multiwoz dataset form Convlab to glue data format
# Cleaning Labels, construct ontology, and divide train, valid, test sets
# 190827, Hwaran Lee

import os
import json
import collections
import argparse
from copy import deepcopy
from utils.create_data import normalize
from utils.fix_label import fix_general_label_error

def construct_ontology(output_dir):

    source_files = ["train.json", "test.json"]
    val_list_file = "valListFile.json"

    ontology_dir = os.path.join(output_dir, "ontology.json")
    ontology_act_dir = os.path.join(output_dir, "ontology_act.json")
    ontology_req_dir = os.path.join(output_dir, "ontology_req.json")

    ## Split train, valid, test dataset
    if not os.path.exists('val.json'):
        print('run split_datset.py')
        import split_dataset
        from zipfile import ZipFile
        for file in ['train.json.zip', 'val.json.zip', 'test.json.zip']:
            with ZipFile(file, 'r') as zipf:
                zipf.extractall()

    ## Load datasets
    train = json.load(open('train.json', 'r'))
    val = json.load(open('val.json','r'))
    test = json.load(open('test.json', 'r'))

    new_train = deepcopy(train)
    new_val = deepcopy(val)
    new_test = deepcopy(test)

    ## Load mapping pair (refer to https://github.com/jasonwu0731/trade-dst/blob/master`/create_data.py)
    fin = open('utils/mapping.pair','r')
    replacements = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

    ## Construct ontology
    ontology = {}
    ontology_act = []
    ontology_req = []

    def normalize_slot_name(slot):
        slot = slot.replace('At', ' at').replace('By', ' by')  # Inform slot
        slot = slot.replace('Addr', 'address').replace('Ref', 'reference') # Request slot
        slot = slot.lower()
        return slot

    for corpus, new_corpus in zip([train, val, test], [new_train, new_val, new_test]):
        for id, dialog in corpus.items():
            for tid, turn in enumerate(dialog['log']):

                if len(turn['metadata']) > 0: # sys turn
                    # extract slot-values in belief states
                    for domain, slot_value in turn['metadata'].items():

                        # Label normalization
                        semi_slot_value = fix_general_label_error(slot_value['semi'])
                        book_slot_value = fix_general_label_error(slot_value['book'])


                        for slot, value in semi_slot_value.items():
                            slot = normalize_slot_name(slot)
                            domain_slot = domain.lower().strip() + '-' + slot.lower().strip()

                            if not domain_slot in ontology:
                                ontology[domain_slot] = []
                            if not value in ontology[domain_slot]:
                                ontology[domain_slot].append(value)

                        for slot, value in book_slot_value.items():
                            if slot == 'booked':
                                continue
                            else:
                                slot = normalize_slot_name(slot)
                                domain_slot = domain.lower().strip() + '-book ' + slot.lower().strip()

                                if not domain_slot in ontology:
                                    ontology[domain_slot] = []
                                if not value in ontology[domain_slot]:
                                    ontology[domain_slot].append(value)

                        new_corpus[id]['log'][tid]['metadata'][domain]['semi'] = semi_slot_value
                        new_corpus[id]['log'][tid]['metadata'][domain]['book'] = book_slot_value

                else: # user turn
                    # extract user dialog act
                    for act, pairs in turn['dialog_act'].items():
                        act = act.lower()

                        if not act in ontology_act:
                            ontology_act.append(act)
                        elif 'request' in act: # DOMAIN-Request
                            domain = act.split('-')[0].lower() # Request SLOT
                            for pair in pairs:
                                slot = normalize_slot_name(pair[0])
                                domain_slot = domain + '-' + slot
                                if not domain_slot in ontology_req:
                                    ontology_req.append(domain_slot)

    ## sort
    ontology_act.sort()
    print('ontology act : %d' % len(ontology_act))
    print(ontology_act)

    ontology_req.sort()
    print('ontology req : %d' % len(ontology_req))
    print(ontology_req)

    ontology = collections.OrderedDict(sorted(ontology.items()))
    print('ontology slots : %d' % len(ontology.keys()))
    print(ontology.keys())
    for slot in ontology.keys():
        ontology[slot].sort()
        print('%s : %d' % (slot, len(ontology[slot])))


    ## Save
    json.dump(ontology, open(ontology_dir, 'w'), indent=4)
    json.dump(ontology_act, open(ontology_act_dir, 'w'), indent=4)
    json.dump(ontology_req, open(ontology_req_dir, 'w'), indent=4)

    json.dump(new_train, open(os.path.join(output_dir, 'train.json'), 'w'), indent=4)
    json.dump(new_val, open(os.path.join(output_dir, 'val.json'), 'w'), indent=4)
    json.dump(new_test, open(os.path.join(output_dir, 'test.json'), 'w'), indent=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    construct_ontology(args.output_dir)
    