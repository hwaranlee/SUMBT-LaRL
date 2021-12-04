import numpy as np
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.enc2dec.decoders import GEN, DecoderRNN
from collections import defaultdict

from convlab.modules.e2e.multiwoz.SUMBT_LaRL.utils.processor import get_sent, get_bert_sent
from tqdm import tqdm, trange
import torch
import pdb
import copy
from convlab.modules.dst.multiwoz.dst_util import init_state

EXCLUDED_DOMAIN = ['bus']
EXCLUDED_DOMAIN_SLOT = ['train-book ticket']

def generate(model, batch, belief_state, gen_type):
    context = batch[0] # user context
    labels = batch[5] # true system response
    num_dialog = labels.size(0)
    num_turn = labels.size(1)
    
    # Forward
    with torch.no_grad():
        outputs = model(batch, n_gpu=1, mode=GEN, gen_type=gen_type)
    
    state_list = None
    if 'output_bs' in outputs:
        state_list = update_states(outputs['output_bs'], model.ontology, belief_state, num_dialog, num_turn)
    
    # move from GPU to CPU
    context = context.cpu().numpy()
    true_labels = labels[:, :, 1:].cpu().numpy() # (batch_size(num_dialog), num_turn, output_seq_len)
    pred_labels = torch.cat(outputs[DecoderRNN.KEY_SEQUENCE], dim=-1) #(ds*ts, output_seq_len)
    pred_labels = pred_labels.view(num_dialog, num_turn, -1).long()
    pred_labels = pred_labels.cpu().numpy()
    
    return context, true_labels, pred_labels, state_list

def task_generate(learning_step, model, dataloader, keys, config, evaluator, bert_tokenizer, dec_vocab, device, num_batch=None, dest_f=None, verbose=True, use_oracle_bs=True):
    def write(msg):
        if msg is None or msg == '':
            return
        if dest_f is None:
            print(msg)
        else:
            dest_f.write(msg + '\n')
            
    model.eval()
    evaluator.initialize()
    generated_dialogs = defaultdict(list)
    de_tknize = lambda x: ' '.join(x)

    belief_state = (init_state())['belief_state']
    
    total_num_dialog = 0
    
    for i, batch in enumerate(tqdm(dataloader, desc="Task Generate")):
        batch = tuple(t.to(device) for t in batch)

        labels = batch[5] # true system response
        num_dialog = labels.size(0)
        num_turn = labels.size(1)
                
        context, true_labels, pred_labels, state_list = generate(model, batch, belief_state, gen_type=config.gen_type)
        
        for did in range(num_dialog):
            for tid in range(num_turn):
                # check padded turn
                if context[did,tid,0] == 0:
                    continue
                
                ctx_str = get_bert_sent(bert_tokenizer, context[did, tid, :])
                true_str = get_sent(dec_vocab, de_tknize, true_labels[did, tid, :])
                pred_str = get_sent(dec_vocab, de_tknize, pred_labels[did, tid, :])
                
                prev_ctx = 'Source context: %s' % ctx_str
                if state_list is None:
                    generated_dialogs[keys[total_num_dialog+did]].append((pred_str, None))
                else:
                    generated_dialogs[keys[total_num_dialog+did]].append((pred_str, state_list[did][tid]))
                evaluator.add_example(true_str, pred_str)

                if verbose and (dest_f is not None):
                    write('%s-prev_ctx = %s' % (keys[total_num_dialog+did], prev_ctx,))
                    write('True: {}'.format(true_str, ))
                    write('Pred: {}'.format(pred_str, ))
                    write('-' * 40)
                    
        total_num_dialog += num_dialog
        
    task_report, success, match = evaluator.evaluateModel(generated_dialogs, learning_step, use_oracle_bs=use_oracle_bs, mode='test') #TODO'valid' vs. 'rollout'
    resp_report, bleu, prec, rec, f1 = evaluator.get_report()
    write(task_report)
    write(resp_report)
    write('Generation Done')
    write('---')
    return success, match, bleu, f1

def update_states(output, ontology, belief_state, num_dialog, num_turn):
        
    # TODO: Cleaning
    new_belief_state_list = []

    for did in range(num_dialog):

        turn_belief_list = []
        for tid in range(num_turn):
    
            new_belief_state = copy.deepcopy(belief_state)
            
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
