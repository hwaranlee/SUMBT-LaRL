import sys, os
sys.path.insert(0, os.path.abspath('.'))

import torch
import json
import math
import pickle
import pdb

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

class LossManager(object):
    def __init__(self, processor):
        self.ontology_domain = processor.ontology_domain
        self.ontology_act = processor.ontology_act
        self.num_domain = len(processor.ontology_domain)
        self.num_act = len(processor.ontology_act)
        self.init()

    def init(self):
        self.total_loss = 0
        self.losses = {}
        self.num_examples = 0
        self.accuracies = {'joint': 0, 'slot': 0, 'joint_rest': 0, 'slot_rest': 0,
                        'num_turn': 0, 'num_slot': 0, 'num_slot_rest': 0,
                        'domain_tp': 0, 'domain_tn': 0, 'domain_pos': 0, 'domain_neg': 0,
                        'act_tp': 0, 'act_tn': 0, 'act_pos': 0, 'act_neg': 0,
                        'domain_tp_slot': torch.zeros(self.num_domain), 'domain_tn_slot': torch.zeros(self.num_domain), 'domain_pos_slot': torch.zeros(self.num_domain), 'domain_neg_slot': torch.zeros(self.num_domain),
                        'act_tp_slot': torch.zeros(self.num_act), 'act_tn_slot': torch.zeros(self.num_act), 'act_pos_slot': torch.zeros(self.num_act), 'act_neg_slot': torch.zeros(self.num_act)
                        }
        
    def add_total_loss(self, loss, ndata):
        self.total_loss += loss.item() * ndata

    def add_loss(self, loss, ndata):
        for key, val in loss.items():
            #if ('loss' in key) or ('acc' in key):
            if not (('pred' in key) or ('hidden' in key) or ('output_bs' in key)):
                if key not in self.losses:
                    self.losses[key] = 0
                self.losses[key] += val.item() * ndata

    def add_num_data(self, ndata):
        self.num_examples += ndata

    def _eval_acc(self, _pred_slot, _labels):
        slot_dim = _labels.size(-1)
        accuracy = (_pred_slot == _labels).view(-1, slot_dim)
        num_turn = torch.sum(_labels[:, :, 0].view(-1) > -1, 0).float()
        num_data = torch.sum(_labels > -1).float()
        # joint accuracy
        joint_acc = sum(torch.sum(accuracy, 1) / slot_dim).float()
        # slot accuracy
        slot_acc = torch.sum(accuracy).float()
        return joint_acc.item(), slot_acc.item(), num_turn.item(), num_data.item()
        
    def eval_all_accs(self, pred_slot, pred_domain, pred_act, labels):
        label_bs, label_domain, label_act = labels

        # 7 domains
        joint_acc, slot_acc, num_turn, num_data = self._eval_acc(pred_slot, label_bs)
        self.accuracies['joint'] += joint_acc
        self.accuracies['slot'] += slot_acc
        self.accuracies['num_turn'] += num_turn
        self.accuracies['num_slot'] += num_data

        # restaurant domain
        joint_acc, slot_acc, num_turn, num_data = self._eval_acc(pred_slot[:,:,20:27], label_bs[:,:,20:27])
        self.accuracies['joint_rest'] += joint_acc
        self.accuracies['slot_rest'] += slot_acc
        self.accuracies['num_slot_rest'] += num_data

        if pred_domain is not None:
            # Domain
            pos = (label_domain == 1)
            neg = (label_domain == 0)
            self.accuracies['domain_pos'] += pos.sum() # num of positive samples
            self.accuracies['domain_neg'] += neg.sum() # num of negative samples
            self.accuracies['domain_tp'] += ((pred_domain == 1).masked_select(pos)).sum() # true positive
            self.accuracies['domain_tn'] += ((pred_domain == 0).masked_select(neg)).sum() # true negative

            self.accuracies['domain_pos_slot'] += pos.sum(0).sum(0).cpu().float()
            self.accuracies['domain_neg_slot'] += neg.sum(0).sum(0).cpu().float()
            self.accuracies['domain_tp_slot'] += ((pred_domain == 1) & pos).sum(0).sum(0).cpu().float() # true positive
            self.accuracies['domain_tn_slot'] += ((pred_domain == 0) & neg).sum(0).sum(0).cpu().float() # true negative

            # Act
            pos = (label_act == 1)
            neg = (label_act == 0)
            self.accuracies['act_pos'] += pos.sum() # num of positive samples
            self.accuracies['act_neg'] += neg.sum() # num of negative samples
            self.accuracies['act_tp'] += ((pred_act == 1).masked_select(pos)).sum() # true positive
            self.accuracies['act_tn'] += ((pred_act == 0).masked_select(neg)).sum() # true negative

            self.accuracies['act_pos_slot'] += pos.sum(0).sum(0).cpu().float()
            self.accuracies['act_neg_slot'] += neg.sum(0).sum(0).cpu().float()
            self.accuracies['act_tp_slot'] += ((pred_act == 1) & pos).sum(0).sum(0).cpu().float() # true positive
            self.accuracies['act_tn_slot'] += ((pred_act == 0) & neg).sum(0).sum(0).cpu().float() # true negative


    def get_accuracies(self):
        acc = []
        acc.append(self.accuracies['joint']/self.accuracies['num_turn'])
        acc.append(self.accuracies['slot']/self.accuracies['num_slot'])
        acc.append(self.accuracies['joint_rest']/self.accuracies['num_turn'])
        acc.append(self.accuracies['slot_rest']/self.accuracies['num_slot_rest'])
        return acc

    def get_precision(self):
        if self.accuracies['domain_pos'] > 0:
            tp = self.accuracies['domain_tp'].item()
            tn = self.accuracies['domain_tn'].item()
            pos = self.accuracies['domain_pos'].item()
            neg = self.accuracies['domain_neg'].item()
            precision_domain = tp/pos if pos > 0 else 0
            recall_domain = tp/(tp+neg-tn) if (tp+neg-tn) > 0 else 0

            tp = self.accuracies['act_tp'].item()
            tn = self.accuracies['act_tn'].item()
            pos = self.accuracies['act_pos'].item()
            neg = self.accuracies['act_neg'].item()
            precision_act = tp / pos if pos > 0 else 0
            recall_act = tp /(tp+neg-tn) if (tp+neg-tn) > 0 else 0

            return precision_domain, precision_act, recall_domain, recall_act
        else:
            return 0, 0, 0, 0
    
    def get_total_loss(self):
        self.total_loss = self.total_loss / self.num_examples
        return self.total_loss
    
    def get_losses(self):
        for key, val in self.losses.items():
            self.losses[key] = val / self.num_examples
        return self.losses

    def print_result(self):

        msg = ''

        if self.accuracies['num_turn'] > 0:
            msg += 'joint acc : slot acc : joint restaurant : slot acc restaurant \n'
            msg += ''.join(['%.5f : '% acc for acc in self.get_accuracies()])
            msg += '\n'

        loss = self.get_losses()
        if 'nll' in loss:
            msg += 'PPL : %.3e \n' % math.exp(loss['nll'])

        for key, val in loss.items():
            if 'loss' in key:
                msg += '%s : %.3e \n' % (key, val)

        if self.accuracies['domain_pos'] > 0:
            tp = self.accuracies['domain_tp'].item()
            tn = self.accuracies['domain_tn'].item()
            pos = self.accuracies['domain_pos'].item()
            neg = self.accuracies['domain_neg'].item()
            msg += 'DOMAIN ACC : DOMAIN PRECISION : DOMAIN RECALL : TP : TN : POS : NEG \n'
            msg += '%.5f : %.5f : %.5f : %d : %d : %d : %d \n' %(
                        (tp+tn)/(pos+neg), tp/pos, tp/(tp+neg-tn), tp, tn, pos, neg)
            
            tp = self.accuracies['act_tp'].item()
            tn = self.accuracies['act_tn'].item()
            pos = self.accuracies['act_pos'].item()
            neg = self.accuracies['act_neg'].item()
            msg += 'ACT ACC : ACT PRECISION : ACT RECALL : TP : TN : POS : NEG \n'
            msg += '%.5f : %.5f : %.5f : %d : %d : %d : %d \n' %(
                        (tp+tn)/(pos+neg), tp/pos, tp/(tp+neg-tn), tp, tn, pos, neg)

            msg += 'DOMAIN : Acc : Precision : Recall : F1 \n'
            for i, domain in enumerate(self.ontology_domain):
                tp = self.accuracies['domain_tp_slot'][i] 
                tn = self.accuracies['domain_tn_slot'][i] 
                pos = self.accuracies['domain_pos_slot'][i] 
                neg = self.accuracies['domain_neg_slot'][i] 
                precision = tp / pos if pos > 0 else 0
                recall = tp / (tp+neg-tn) if (tp+neg-tn) > 0 else 0
                if tp == 0:
                    msg += '%s : %.5f : %.5f : %.5f : %.5f : tp(%d) tn(%d) pos(%d) neg(%d) \n' % (domain, 0, 0, 0, 0, tp, tn, pos, neg)
                elif pos == 0 or (tp+neg-tn) == 0:
                    msg += '%s : ??? ' % domain
                else:                 
                    msg += '%s : %.5f : %.5f : %.5f : %.5f \n' % (domain, (tp+tn)/(pos+neg), precision, recall, 2/(1/precision + 1/recall))

            msg += 'ACT : Acc : Precision : Recall : F1 \n'
            for i, domain in enumerate(self.ontology_act):
                tp = self.accuracies['act_tp_slot'][i] 
                tn = self.accuracies['act_tn_slot'][i] 
                pos = self.accuracies['act_pos_slot'][i] 
                neg = self.accuracies['act_neg_slot'][i] 
                precision = tp / pos  if pos > 0 else 0
                recall = tp / (tp+neg-tn) if (tp+neg-tn) > 0 else 0
                if tp == 0:
                    msg += '%s : %.5f : %.5f : %.5f : %.5f : tp(%d) tn(%d) pos(%d) neg(%d) \n' % (domain, 0, 0, 0, 0, tp, tn, pos, neg)
                elif pos == 0 or (tp+neg-tn) == 0:
                    msg += '%s : ??? ' % domain
                else:                 
                    msg += '%s : %.5f : %.5f : %.5f : %.5f \n' % (domain, (tp+tn)/(pos+neg), precision, recall, 2/(1/precision + 1/recall))
        return msg


def save_configure(args, labels, processor):
    with open(os.path.join(args.output_dir, "config.json"),'w') as outfile:
        data = {"config": vars(args),
                "labels": labels,
                "ontology": processor.ontology,
                "vocab_dict": processor.vocab_dict,
                }
        json.dump(data, outfile, indent=4)
        

def get_key_list(data):
    # make keys 
    keys=[]
    for ex in data:
        key = ex['guid'].split('-')[1]
        if len(keys) == 0 or keys[-1] != key:
            keys.append(key)
    return keys
        