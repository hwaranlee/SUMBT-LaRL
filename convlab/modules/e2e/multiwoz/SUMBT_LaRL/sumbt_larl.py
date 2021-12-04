
import os.path
import copy
import json
from argparse import Namespace
import urllib.request
import tarfile
import random

import torch
import convlab
from convlab.lib import logger
from convlab.modules.dst.multiwoz.dst_util import init_state, init_belief_state
from convlab.modules.policy.system.policy import SysPolicy
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.model import SUMBTLaRL as model
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.enc2dec.decoders import GEN, TEACH_FORCE
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.utils.processor import (Processor, truncate_seq_pair,
                                                                     get_label_embedding, get_sent, get_bert_sent,
                                                                     UNK, PAD, BOS, EOS )
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.utils.db_utils import make_db_feature, populate_template, get_active_domain, query
from pytorch_pretrained_bert.tokenization import BertTokenizer

#logger = logger.getLogger(__name__)

DEFAULT_MODEL_DIR = ".//convlab/modules/e2e/multiwoz/SUMBT_LaRL/models"
DEFAULT_DOWNLOAD_DIR = ".//convlab/modules/e2e/multiwoz/SUMBT_LaRL/models"
THRESHOLD=0.5
EXCLUDED_DOMAIN = ['bus']
EXCLUDED_DOMAIN_SLOT = ['train-book ticket']

class ConvLabProcessor(Processor):
    def __init__(self, config, ontology, labels, vocab_dict):
        self.input_template = {'guid':'', 'prev_sys': '', 'user': '', 'user_delex': '', 'sys': '', 'sys_delex':'',
                               'bs': {}, 'user_req': {}, 'user_gen': {},
                               'source':{}, 'kb':'', 'db':{}}

        self.ontology = ontology
        self.ontology_domain = labels[1]
        self.ontology_act = labels[2]
        self.target_slot = list(self.ontology.keys())

        # Policy and Decoder (For MultiWOZ)
        self.bs_size = len(self.ontology)
        self.db_size = config.db_size
        self.vocab_dict = vocab_dict
        self.vocab = list(self.vocab_dict)
        self.unk_id = self.vocab_dict[UNK]
        self.pad_id = self.vocab_dict[PAD]
        self.dec_max_seq_len = config.dec_max_seq_len

    def turn_preprocess(self, user_utter, sys_utter, max_seq_length, max_turn_length, tokenizer):
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


class SUMBT_LaRL(SysPolicy):
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, download_path=None):
        SysPolicy.__init__(self)

        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("device: {}".format(self.device))

        if not (os.path.isfile(os.path.join(model_dir, 'config.json')) and os.path.isfile(os.path.join(model_dir, 'pytorch_model.bin'))):
            if not download_path:
                raise Exception("No model for SUMBT_LARL is specified!")

            else:
                if not os.path.exists(DEFAULT_DOWNLOAD_DIR):
                    os.makedirs(DEFAULT_DOWNLOAD_DIR, exist_ok=True)

                    model_name = download_path.split('/')[-1]
                    download_dir = os.path.join(DEFAULT_DOWNLOAD_DIR, model_name)
                    logger.info("Download model from %s to %s " % (download_path,DEFAULT_DOWNLOAD_DIR))
                    urllib.request.urlretrieve(download_path, download_dir)

                    self.model_dir = os.path.join(DEFAULT_DOWNLOAD_DIR, model_name.replace('.tar.gz', ''))
                    tar = tarfile.open(download_dir, "r:gz")
                    tar.extractall(path=self.model_dir)
                    tar.close()
                    logger.info("... extracted to %s " % self.model_dir)

        configs = json.load(open(os.path.join(self.model_dir, 'config.json')))
        self.config = Namespace(**configs['config'])
        labels = configs['labels']
        ontology = configs['ontology']
        vocab_dict = configs['vocab_dict']

        self.processor = ConvLabProcessor(self.config, ontology, labels, vocab_dict)

        vocab_dir = os.path.join(self.config.bert_dir, '%s-vocab.txt' % self.config.bert_model)
        if not os.path.exists(vocab_dir):
            download_bert_models(self.config.bert_model, self.config.bert_dir)

        self.tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case = self.config.do_lower_case)

        self.de_tknize = lambda x: ' '.join(x)

        self.model = model(args=self.config, processor=self.processor,
                           labels=labels, device=self.device)
        self.load_model()
        self.model.eval()

        self.max_seq_length = self.config.max_seq_length # max encoding sequcne
        self.threshold = THRESHOLD

        self.init_session()

    def load_model(self):
        model_file = os.path.join(self.model_dir, 'pytorch_model.bin')
        if not os.path.isfile(model_file):
            raise Exception("No model for SUMBT_LaRL is specified!")
        logger.info('Load SUMBT_LaRL from %s' % model_file)

        ptr_model = torch.load(model_file)
        self.model.load_state_dict(ptr_model)
        self.model.to(self.device)

    def reset_seed(self, seed=42):
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.model.reset_seed(seed)

    def init_session(self):
        """Init the Tracker to start a new session."""
        self.state = init_state()
        self.prev_active_domain = None
        self.context_input_ids = None
        self.context_input_len = None
        self.reset_seed(42)

    def predict(self, user_utter):

        if not isinstance(user_utter, str):
            raise Exception('Expected user_act to be <class \'str\'> type, but get {}.'.format(type(user_utter)))

        # preprocessing
        if len(self.state['history']) == 0: # first turn
            sys_utter = None
            self.state['history'].append([None])
        else:
            sys_utter = self.state['history'][-1][0]

        input_ids, input_len = self.processor.turn_preprocess(user_utter, sys_utter,
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

        # forward SUMBT
        dst_results = self.model.forward_sumbt(input_ids, input_len)
        output_bs = dst_results['output_bs']
        out_domain = torch.sigmoid(dst_results['hidden_domain'][0, -1,:]).squeeze()
        out_act = torch.sigmoid(dst_results['hidden_act'][0, -1,:]).squeeze()

        # update state
        prev_state = self.state
        self.state = self._update_state(output_bs, out_domain, out_act, prev_state)
        self.state['history'][-1].append(user_utter)

        # query and make DB feature
        db_feat, _, _ = make_db_feature(self.state['belief_state'], mode='predict')
        db_feat = torch.tensor(db_feat, dtype=out_domain.dtype).to(self.device)
        db_feat = db_feat.view(1,1,-1)

        # make policy input (select the last turn input vector)
        dst_results['hidden_utt'] = dst_results['hidden_utt'][-1,:].unsqueeze(0)
        dst_results['hidden_act'] = dst_results['hidden_act'][:,-1,:].unsqueeze(1)
        dst_results['hidden_domain'] = dst_results['hidden_domain'][:,-1,:].unsqueeze(1)

        for i, val in enumerate(dst_results['output_bs']):
            dst_results['output_bs'][i] = val[:,-1,:].unsqueeze(1)

        # forward LaRL
        outputs = self.model.forward_policy(output_ids=None, db_feat=db_feat, dst_results=dst_results,
                                            mode=GEN, gen_type=self.config.gen_type)


        # get response text
        pred_labels = torch.cat(outputs['sequence'], dim=-1) #(ds*ts, output_seq_len)
        pred_labels = pred_labels.squeeze().long()
        pred_labels = pred_labels.cpu().numpy()
        delex_sys_resp = get_sent(self.processor.vocab, self.de_tknize, pred_labels)

        # lexicalize
        top_results, num_results, active_domain = self._query_result(out_domain, prev_state['belief_state'], self.state['belief_state'])
        sys_resp = populate_template(delex_sys_resp, top_results, num_results, self.state['belief_state'], active_domain)


        # post-process
        logger.state(f"1st Lexicalized system utterance: {sys_resp}")
        self.cur_dom = active_domain
        whole_kb = top_results

        try:
            logger.state(f"DB: {top_results[self.cur_dom]}")
            sys_resp = self.postprocess(sys_resp, top_results[self.cur_dom], whole_kb[self.cur_dom])
        except KeyError:
            sys_resp = self.postprocess(sys_resp, [], [])

        logger.state(f"2nd Lexicalized system utterance: {sys_resp}")

        # inactive domain: postprocess heuristic
        sys_resp = sys_resp.replace('[hotel_postcode]', 'cb21ab')

        # fix entrance fee
        sys_resp = sys_resp.replace('any information on the entrance fee', 'entrance fee is unknown pound.')
        sys_resp = sys_resp.replace('entrance fee is not listed', 'entrance fee is unknown pound.')

        # update state sys response
        self.state['history'].append([sys_resp])
        self.prev_active_domain = active_domain

        #logger.state(f"Dialog state: {self.state}")
        pprint_logger_state(self.state)
        logger.db(f"Active domain: {active_domain}")
        logger.db(f"Top DB results: {top_results}")
        logger.db(f"Number of DB results: {num_results}")

        logger.delex(f"Delexicalized system utterance: {delex_sys_resp}")

        return sys_resp

    def _query_result(self, out_domain, prev_belief_state, belief_state):

        vidx = torch.argmax(out_domain)
        active_domain_model = self.processor.ontology_domain[vidx]
        active_domain_rule = get_active_domain(self.prev_active_domain, prev_belief_state, belief_state)

        out_domain_above_th = torch.ge(out_domain, THRESHOLD)
        active_domain_list = []
        for i, val in enumerate(out_domain_above_th):
            if val == 1:
                active_domain_list.append(self.processor.ontology_domain[i])

        active_domain = None
        if active_domain_list:
            if active_domain_rule is not None:
                active_domain_list.append(active_domain_rule)

            active_domain_list = list(set(active_domain_list))

            if len(active_domain_list) == 1:
                active_domain = active_domain_list[0]
            else:
                active_domain = active_domain_model
        else:
            if active_domain_rule is not None:
                active_domain = active_domain_rule
            else:
                active_domain = active_domain_model

        domains_ex = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
        top_results_temp = {}
        num_results = {}
        for domain in domains_ex:
            if domain != 'train':
                entities, _ = query(domain, belief_state[domain]['semi'])
            else:
                entities, _ = query(domain, belief_state[domain]['semi'])
            num_results[domain] = len(entities)
            if len(entities) > 0:
                if domain == 'train':
                    if belief_state[domain]['semi']['leaveAt'] != '':
                        top_results_temp[domain] = sorted(entities, key=lambda k: k['leaveAt'])
                    else:
                        top_results_temp[domain] = sorted(entities, key=lambda k: k['arriveBy'], reverse=True)
                top_results_temp[domain] = entities[0]
            else:
                top_results_temp[domain] = {}

        #if active_domain is not None and active_domain in num_results:
        #    num_results = num_results[active_domain]
        #else:
        #    num_results = 0

        top_results = {}
        if active_domain is not None:
            if len(active_domain_list) == 1:
                top_results.update({active_domain: top_results_temp[active_domain]})
            else:
                for domain_can in active_domain_list:
                    top_results.update({domain_can: top_results_temp[domain_can]})

        return top_results, num_results, active_domain


    def _update_state(self, output, out_domain, out_act, prev_state):

        # TODO: Cleaning

        new_belief_state = copy.deepcopy(prev_state['belief_state'])
        new_request_state = copy.deepcopy(prev_state['request_state'])
        user_action = {}

        # update belief state and Inform action
        for sidx, domain_slot in enumerate(self.processor.ontology.keys()):

            domain = domain_slot.split('-')[0].strip()
            if domain in EXCLUDED_DOMAIN or domain_slot in EXCLUDED_DOMAIN_SLOT:
                continue

            slot = slot_normlization(domain_slot.split('-')[1]) # normalize {pricerange, leaveat, arriveby}

            vidx = torch.argmax(output[sidx][0,-1,:])
            value = self.processor.ontology[domain_slot][vidx]

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
                    if (domain + "-Inform(BS)") not in user_action:
                        user_action[domain + "-Inform(BS)"] = []
                    if [slot, value] not in user_action[domain + "-Inform(BS)"]:
                        user_action[domain + "-Inform(BS)"].append([slot, value])

            else:
                # update belief
                prev_value = new_belief_state[domain]['semi'][slot]
                if value != prev_value:
                    new_belief_state[domain]['semi'][slot] = value

                    # update action
                    domain, slot = act_domain_slot_normalization(domain, slot)
                    if (domain + "-Inform(BS)") not in user_action:
                        user_action[domain + "-Inform(BS)"] = []
                    if [slot, value] not in user_action[domain + "-Inform(BS)"]:
                        user_action[domain + "-Inform(BS)"].append([slot, value])

        # update request state and Request action
        for sidx, domain_slot in enumerate(self.processor.ontology_act):
            act = domain_slot.split('-')[0]
            request = domain_slot.split('-')[0] == 'request'
            inform = domain_slot.split('-')[0] == 'inform'

            # domain_slot preprocessing
            domain = domain_slot.split('-')[0].strip()
            if act == 'general':
                # find value
                if out_act[sidx] > self.threshold:
                    if not domain_slot in user_action:
                        user_action[domain_slot] = []
                    user_action[domain_slot].append(["none", "none"])

            elif request:
                if out_act[sidx] > self.threshold:
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
                    else:
                        new_request_state[domain][slot] += 1

            elif inform:
                if domain in EXCLUDED_DOMAIN or domain_slot in EXCLUDED_DOMAIN_SLOT:
                    continue
                if out_act[sidx] > self.threshold:
                    slot = slot_normlization(domain_slot.split('-')[1])  # normalize {pricerange, leaveat, arriveby}
                    domain, slot = act_domain_slot_normalization(domain, slot)
                    if (domain + "-Inform") not in user_action:
                        user_action[domain + "-Inform"] = []
                    if [slot, value] not in user_action[domain + "-Inform"]:
                        user_action[domain + "-Inform"].append([slot, value])

        # update state
        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state

        # update user act
        new_state['user_action'] = user_action

        return new_state

    def postprocess(self, out_text, kb_results, whole_kb):
        # heuristics
        if 'center of town' in out_text:
            out_text = out_text.replace('center of town', 'centre')
        if 'south part of town' in out_text:
            out_text = out_text.replace('south part of town', 'south')
        if 'no entrance fee' in out_text:
            out_text = out_text.replace('no entrance fee', 'free')
        if 'free to enter' in out_text:
            out_text = out_text.replace('free to enter', 'free')
        if 'No entrance fee' in out_text:
            out_text = out_text.replace('No entrance fee', 'free')

        # added
        if 'centre part of town' in out_text:
            out_text = out_text.replace('centre part of town', 'centre')
        if 'east part of town' in out_text:
            out_text = out_text.replace('east part of town', 'east')
        if 'west part of town' in out_text:
            out_text = out_text.replace('west part of town', 'west')
        if 'north part of town' in out_text:
            out_text = out_text.replace('north part of town', 'north')
        if 'centre area' in out_text:
            out_text = out_text.replace('centre area', 'centre')
        if 'centre of town' in out_text:
            out_text = out_text.replace('centre of town', 'centre')
        if 'expensively priced' in out_text:
            out_text = out_text.replace('expensively priced', 'expensive')
        if 'cheaply priced' in out_text:
            out_text = out_text.replace('cheaply priced', 'cheap')
        if 'moderately priced' in out_text:
            out_text = out_text.replace('moderately priced', 'moderate')
        if 'train id' in out_text:
            out_text = out_text.replace('the train id is ', 'I have a train ')
        if 'gbp' in out_text:
            out_text = out_text.replace('gbp', 'pounds')

        sv = ['reference', 'trainid', 'postcode', 'phone', 'address', 'name', 'duration', 'price']
        slots = ['[' + self.cur_dom + '_' + s + ']' for s in sv]
        default_value = {'reference': '00000000', 'trainid': 'tr7075', 'postcode': 'cb21ab', 'phone': '01223351880', 'name': 'error',
                        'address': "Hills Rd , Cambridge", 'duration': '15:00', 'price': '9.80'}

        kb_results = self.convert_kb(kb_results)

        for slot, s in zip(slots, sv):
            t = s

            if slot in slots:
                if out_text.count(slot) > 1:
                    print(slot)
                    try:
                        if len(kb_results) > 1:
                            out_tok = []
                            tmp = copy.deepcopy(out_text).split(' ')
                            k = 0
                            for tok in tmp:
                                if tok == slot:
                                    out_tok.append(self.convert_kb(whole_kb[k])[t])
                                    k += 1
                                else:
                                    out_tok.append(tok)

                                out_text = ' '.join(out_tok)
                    except:
                        out_text = out_text.replace(slot, default_value[t])

                else:
                    try:
                        if slot == '[taxi_phone]':
                            out_text = out_text.replace(slot, ''.join(kb_results['taxi_phone']))
                        else:
                            out_text = out_text.replace(slot, kb_results[t])
                    except:
                        if '[' in out_text:
                            logger.state(f"ERR : {t}")
                        out_text = out_text.replace(slot, default_value[t])

        return out_text.strip()

    def convert_kb(self, kb_results):
        new_kb = {}
        for key in kb_results:

            value = kb_results[key]
            if key == 'arriveBy':
                key = 'arriveby'
            elif key == 'leaveAt':
                key = 'leaveat'
            elif key == 'trainID':
                key = 'trainid'
            elif key == 'Ref':
                key = 'reference'
            elif key == 'address':
                key = 'address'
            elif key == 'duration':
                key = 'duration'
            elif key == 'postcode':
                key = 'postcode'
            new_kb[key] = value

        return new_kb


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

def act_domain_slot_normalization(domain, slot):
    # Act slots
    # slots defined self.processor.ontology and self.processor.ontology_request should be matched to values of REF_USR_DA[domain]
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

def  pprint_logger_state(state):

    logger.state(f"Dialog state:")
    logger.state(f" user_action: {state['user_action']}")
    logger.state(f" belief_state: ")
    for key, val in state['belief_state'].items():
        logger.state(f"  {key}: {val}")
    logger.state(f" reqest_state: {state['request_state']}")
    logger.state(f" history: ")
    for turn in state['history']:
        logger.state(f"  {turn}")


def download_bert_models(bert_model, pretrained_bert_dir):
    if not os.path.exists(pretrained_bert_dir):
        os.makedirs(pretrained_bert_dir, exist_ok=True)

    # download vocab file
    PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
    }
    logger.info("Downloading pretrained vocab %s to %s" %(bert_model, pretrained_bert_dir))
    urllib.request.urlretrieve(PRETRAINED_VOCAB_ARCHIVE_MAP[bert_model], os.path.join(pretrained_bert_dir, '%s-vocab.txt' % bert_model))

    # download model file
    PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
    }
    logger.info("Downloading pretrained model %s to %s" %(bert_model, pretrained_bert_dir))
    urllib.request.urlretrieve(PRETRAINED_MODEL_ARCHIVE_MAP[bert_model], os.path.join(pretrained_bert_dir, '%s.model' % bert_model))

