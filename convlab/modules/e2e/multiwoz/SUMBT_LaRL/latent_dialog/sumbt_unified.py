import os.path
import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
from torch.nn import BCEWithLogitsLoss

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl(sequence length) * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores

class BeliefTracker(nn.Module):
    ### Domain classifier, Act classifier (general, inform-slot, domainuest-slot)
    def __init__(self, args, labels, device, model_dir=None):
        super(BeliefTracker, self).__init__()

        self.model_dir = model_dir if model_dir else None
        if hasattr(args, 'labels'):
            labels = args.labels

        # parsing configure
        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length

        self.num_labels = [len(label) for label in labels[0]]  # number of slot-values in each slot-type
        self.num_domain = len(labels[1])
        self.num_act = len(labels[2])
        self.num_slots = len(self.num_labels)
        self.attn_head = args.attn_head
        self.device = device

        self.get_utt_from_ptr = args.get_utt_from_ptr

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, '%s.model'%args.bert_model)
        )    
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        self.dropout_prob = self.hidden_dropout_prob # args.dropout
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.parameters():
                p.requires_grad = False
        
        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = deepcopy(self.utterance_encoder)
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in self.num_labels])

        ### Attention layers
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=self.hidden_dropout_prob)

        ### RNN Belief Tracker
        self.nbt = None
        if args.task_name.find("gru") != -1:
            self.nbt = nn.GRU(input_size=self.bert_output_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.rnn_num_layers,
                              dropout=self.dropout_prob,
                              batch_first=True)
            self.init_parameter(self.nbt)
        elif args.task_name.find("lstm") != -1:
            self.nbt = nn.LSTM(input_size=self.bert_output_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.rnn_num_layers,
                               dropout=self.dropout_prob,
                               batch_first=True)
            self.init_parameter(self.nbt)
        if not self.zero_init_rnn:
            self.rnn_init_linear = nn.Sequential(
                                nn.Linear(self.bert_output_dim, self.hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(self.dropout_prob)
                                )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        self.common_linear = nn.Sequential(
            nn.Linear(self.bert_output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        ### Domain classifier
        self.domain_clf = nn.Sequential(
            #nn.Linear(self.hidden_dim, self.hidden_dim),
            #nn.LayerNorm(self.hidden_dim),
            #nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_domain)
        )

        ### Act classifier: general, inform-slot, domainuest-slot
        self.act_clf = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_act)
        )

        ### Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        if args.diff_pos_weight:
            self.domain_pos_weight = torch.Tensor(DOMAIN_POS_WEIGHT)
            self.act_pos_weight = torch.Tensor(ACT_POS_WEIGHT)
            
            for i, x in enumerate(DOMAIN_POS_WEIGHT):
                x = round(x / args.domain_pos_weight)
                self.domain_pos_weight[i] = x if x > 1 else 1
            for i, x in enumerate(ACT_POS_WEIGHT):
                x = round(x / args.act_pos_weight)
                self.act_pos_weight[i] = x if x > 1 else 1
        else:
            self.domain_pos_weight = torch.ones([self.num_domain])*args.domain_pos_weight
            self.act_pos_weight = torch.ones([self.num_act])*args.act_pos_weight

        self.domain_pos_weight = torch.Tensor(self.domain_pos_weight)
        self.act_pos_weight = torch.Tensor(self.act_pos_weight)
        
        self.bce_domain = BCEWithLogitsLoss(pos_weight = self.domain_pos_weight)
        self.bce_act = BCEWithLogitsLoss(pos_weight = self.act_pos_weight)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(self.device)
        slot_mask = slot_ids > 0
        hid_slot, _ = self.sv_encoder(slot_ids.view(-1, self.max_label_length),
                                   slot_type_ids.view(-1, self.max_label_length),
                                   slot_mask.view(-1, self.max_label_length),
                                           output_all_encoded_layers=False)
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(self.device)
            label_mask = label_id > 0
            hid_label, _ = self.sv_encoder(label_id.view(-1, self.max_label_length),
                                           label_type_ids.view(-1, self.max_label_length),
                                           label_mask.view(-1, self.max_label_length),
                                           output_all_encoded_layers=False)
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")

    def _make_aux_tensors(self, ids, len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for i in range(len.size(0)):
            for j in range(len.size(1)):
                if len[i,j,0] == 0: # padding
                    break
                elif len[i,j,1] > 0: # escape only text_a case
                    start = len[i,j,0]
                    ending = len[i,j,0] + len[i,j,1]
                    token_type_ids[i, j, start:ending] = 1
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    def forward(self, input_ids, input_len, labels=None, n_gpu=1, forward_utterance_encoder=False, target_slot=None):

        results = {}

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0) # dialog size
        ts = input_ids.size(1) # turn size
        bs = ds*ts
        slot_dim = len(target_slot)

        # Utterance encoding
        token_type_ids, attention_mask = self._make_aux_tensors(input_ids, input_len)

        hidden_bert, hidden_utt = self.utterance_encoder(input_ids.view(-1, self.max_seq_length),
                                           token_type_ids.view(-1, self.max_seq_length),
                                           attention_mask.view(-1, self.max_seq_length),
                                           output_all_encoded_layers=False)
        """
        # utterance encoding using pretrained bert (not fine-tuned)
        _, hidden_utt_ptr = self.sv_encoder(input_ids.view(-1, self.max_seq_length),
                                           token_type_ids.view(-1, self.max_seq_length),
                                           attention_mask.view(-1, self.max_seq_length),
                                           output_all_encoded_layers=False)
        
        results['hidden_utt'] = hidden_utt_ptr if self.get_utt_from_ptr else hidden_utt
        """
        results['hidden_utt'] = hidden_utt
        if forward_utterance_encoder:
            return results

        hidden_bert = torch.mul(hidden_bert, attention_mask.view(-1, self.max_seq_length, 1).expand(hidden_bert.size()).float())
        hidden = hidden_bert.repeat(slot_dim, 1, 1)          #[(slot_dim*ds*ts), bert_seq, hid_size]

        hid_slot = self.slot_lookup.weight[target_slot, :]            # Select target slot embedding
        hid_slot = hid_slot.repeat(1, bs).view(bs*slot_dim, -1)       #[(slot_dim*ds*ts), bert_seq, hid_size]

        # Attended utterance vector
        hidden = self.attn(hid_slot, hidden, hidden, mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1))
        hidden = hidden.squeeze() # [slot_dim*ds*ts, bert_dim]
        hidden = hidden.view(slot_dim, ds, ts, -1).view(-1,ts,self.bert_output_dim)

        # NBT
        if self.zero_init_rnn:
            h = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim).to(self.device)  # [1, slot_dim*ds, hidden]
        else:
            h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn_num_layers, 1, 1)
            h = self.rnn_init_linear(h)
            
        if isinstance(self.nbt, nn.GRU):
             rnn_out, _ = self.nbt(hidden, h)  # [slot_dim*ds, turn, hidden]
        elif isinstance(self.nbt, nn.LSTM):
            c = torch.zeros(self.rnn_num_layers, input_ids.shape[0]*slot_dim, self.hidden_dim).to(self.device)  # [1, slot_dim*ds, hidden]
            rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
        rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out)))

        hidden = rnn_out.view(slot_dim, ds, ts, -1)

        hidden_common = self.common_linear(hidden_utt)
        
        # Domain classifier
        hidden_domain = self.domain_clf(hidden_common).view(ds, ts, self.num_domain)

        # Act classifier
        hidden_act = self.act_clf(hidden_common).view(ds, ts, self.num_act)

        # Calculate Loss
        loss_domain = 0
        loss_act = 0
        if labels is not None:
            _, label_domain, label_act = labels
            mask = (label_act == -1)
            num_data = (label_act != -1).sum()
            hidden_act = hidden_act.masked_fill(mask, 1)
            loss_act += self.bce_act(hidden_act, label_act.masked_fill(mask, 1).float())
            pred_act = (torch.sigmoid(hidden_act) > 0.5).long()
            acc_act = (pred_act == label_act).sum().float() / num_data

            mask = (label_domain == -1)
            num_data = (label_domain != -1).sum()
            hidden_domain = hidden_domain.masked_fill(mask, 1)
            loss_domain += self.bce_domain(hidden_domain, label_domain.masked_fill(mask, 1).float())
            pred_domain = (torch.sigmoid(hidden_domain) > 0.5).long()
            acc_domain = (pred_domain == label_domain).sum().float() / num_data

        # Label (slot-value) encoding
        loss_bs = 0 # belief state
        loss_slot =[]
        pred_slot = []
        output = []

        for s, slot_id in enumerate(target_slot): ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds*ts*num_slot_labels, -1)
            _hidden = hidden[s,:,:,:].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(ds*ts*num_slot_labels, -1)
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                label_bs = labels[0]
                _loss = self.nll(_dist.view(ds*ts, -1), label_bs[:,:,s].view(-1))
                loss_slot.append(_loss.item())
                loss_bs += _loss

        # Hidden vectors
        results['output_bs'] = output
        results['hidden_domain'] = hidden_domain
        results['hidden_act'] = hidden_act

        if labels is None:    
            return results

        else:
            # calculate joint accuracy
            label_bs = labels[0]
            pred_slot = torch.cat(pred_slot, 2)
            accuracy = (pred_slot == label_bs).view(-1, slot_dim)
            acc_slot = torch.sum(accuracy,0).float() \
                    / torch.sum(label_bs.view(-1, slot_dim) > -1, 0).float()
            acc = sum(torch.sum(accuracy, 1) / slot_dim).float() \
                / torch.sum(label_bs[:,:,0].view(-1) > -1, 0).float() # joint accuracy

            # append loss and accuracies
            results['loss_bs'] = loss_bs
            results['acc'] = acc
            #results['loss_slot'] = loss_slot
            #results['acc_slot'] = acc_slot
            results['pred_slot'] = pred_slot
            results['loss_domain'] = loss_domain
            results['acc_domain'] = acc_domain
            results['pred_domain'] = pred_domain
            results['loss_act'] = loss_act
            results['acc_act'] = acc_act
            results['pred_act'] = pred_act

            if n_gpu > 1:
                for key, val in results.items():
                    results[key] = val.unsqueeze(0)

            return results

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)


