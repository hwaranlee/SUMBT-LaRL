import os.path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.sumbt_unified import BeliefTracker, BertForUtteranceEncoding
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.policy_models import SysPerfectBD2Cat as SysPolicy
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.enc2dec.decoders import GEN, TEACH_FORCE

class SUMBTLaRL(nn.Module):
    def __init__(self, args, processor, labels, device):
        super(SUMBTLaRL, self).__init__()

        # pretraining option
        # if all variables below are false then end-to-end training
        self.pretrain = args.pretrain             # pretraining each module simultaneuosly
        self.pretrain_sumbt = args.pretrain_sumbt # pretraining only sumbt
        self.pretrain_larl = args.pretrain_larl  # pretraining only larl

        if (self.pretrain_sumbt and self.pretrain_larl) == True:
            raise ValueError("set --pretrain")

        self.ontology = processor.ontology
        self.none_idx = []
        for vals in self.ontology.values():
            for idx, val in enumerate(vals):
                if val == 'none':
                    self.none_idx.append(idx)

        ### Word-DST
        self.dst = BeliefTracker(args, labels, device)
        args.utt_emb_size = self.dst.bert_output_dim
        self.fix_utterance_encoder = args.fix_utterance_encoder

        ### Word-Policy
        self.policy = SysPolicy(args, processor, device)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):
        self.dst.initialize_slot_value_lookup(label_ids, slot_ids)

    def initialize_sumbt(self, model_dir):
        # Load the pretrained model
        output_model_file = os.path.join(model_dir, "pytorch_model.bin")
        ptr_model = torch.load(output_model_file)
        model_dict = self.dst.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k.replace('dst.', ''): v for k, v in ptr_model.items() if k.replace('dst.', '') in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        assert len(pretrained_dict) == len(model_dict)
        # 3. load the new state dict
        self.dst.load_state_dict(pretrained_dict)

    def initialize_larl(self, model_dir):
        # Load the pretrained model
        output_model_file = os.path.join(model_dir, "pytorch_model.bin")
        ptr_model = torch.load(output_model_file)
        model_dict = self.policy.state_dict()
        pretrained_dict = {k.replace('policy.', ''): v for k, v in ptr_model.items() if k.replace('policy.', '') in model_dict}
        model_dict.update(pretrained_dict)
        assert len(pretrained_dict) == len(model_dict)
        self.policy.load_state_dict(pretrained_dict)

    def make_bs_feature(self, probs):
        # output is a vector of size slot numbers
        output = []
        for i, prob in enumerate(probs):
            prob = F.softmax(prob, dim=-1)
            output.append(1-prob[:,:,self.none_idx[i]].unsqueeze(-1)) # prob(!none)
            output.append(prob[:,:,self.none_idx[i]].unsqueeze(-1)) # prob(none)
        output = torch.cat(output, dim=-1)
        return output

    def make_bs_feature_from_label(self, labels):
        # output is a vector of size slot numbers
        output = []
        for i in range(labels.size(2)):
            output.append((1- (labels[:,:,i] == self.none_idx[i])).unsqueeze(-1)) # none = 0, value = 1
            output.append((labels[:,:,i] == self.none_idx[i]).unsqueeze(-1)) # none = 0, value = 1
        output = torch.cat(output, dim=-1)
        return output

    def forward(self, batch, n_gpu=1, target_slot=None, mode=TEACH_FORCE, gen_type='greedy'):
        input_ids, input_len, label_bs_ids, label_domain_ids, label_act_ids, output_ids, db_feat = batch
        labels = [label_bs_ids, label_domain_ids, label_act_ids]
        output = {}

        dst_results = self.forward_sumbt(input_ids, input_len, labels=labels, n_gpu=n_gpu)
        if not self.pretrain_larl:
            for key, val in dst_results.items():
                if ('loss' in key) or ('acc' in key) or ('pred' in key) or ('output_bs' in key):
                    output[key] = val

        if self.pretrain_sumbt:
            return output

        policy_results = self.forward_policy(dst_results, db_feat, output_ids, labels, mode, gen_type)

        for key, val in policy_results.items():
            output[key] = val

        if not mode == GEN:
            policy_loss = self.policy.valid_loss(policy_results, batch_cnt=None)
            output['loss_policy'] = policy_loss

        return output

    def forward_sumbt(self, input_ids, input_len, labels=None, n_gpu=1, target_slot=None):

        if self.pretrain_larl: # not update belows during pretraining larl
            dst_results = self.dst(input_ids, input_len, labels=labels, n_gpu=n_gpu, target_slot=target_slot, forward_utterance_encoder=True)
        else:
            dst_results = self.dst(input_ids, input_len, labels=labels, n_gpu=n_gpu, target_slot=target_slot)

        if self.fix_utterance_encoder or self.pretrain_larl:
            dst_results['hidden_utt'] = dst_results['hidden_utt'].detach()

        return dst_results

    def forward_policy(self, dst_results, db_feat, output_ids=None, labels=None, mode=TEACH_FORCE, gen_type='greedy'):
        policy_input = {}
        policy_input['hidden_utt'] = dst_results['hidden_utt']

        if self.pretrain or self.pretrain_larl:
            label_bs_ids, label_domain_ids, label_act_ids = labels
            policy_input['bs'] = self.make_bs_feature_from_label(label_bs_ids).float()
            policy_input['act'] = torch.cat([label_domain_ids, label_act_ids], dim=-1).float()
        else:
            policy_input['bs'] = self.make_bs_feature(dst_results['output_bs'])
            hidden_domain = torch.sigmoid(dst_results['hidden_domain'])
            hidden_act = torch.sigmoid(dst_results['hidden_act'])
            policy_input['act'] = torch.cat([hidden_domain, hidden_act], dim=-1)

        policy_input['db'] = db_feat
        policy_input['output_ids'] = output_ids

        policy_results = self.policy(policy_input, mode=mode, gen_type=gen_type)

        return policy_results

    def forward_rl(self, batch, n_gpu=1, max_words=None, temp=0.1):

        input_ids, input_len, label_bs_ids, label_domain_ids, label_act_ids, _, db_feat = batch
        labels = [label_bs_ids, label_domain_ids, label_act_ids]

        dst_results = self.forward_sumbt(input_ids, input_len, labels=labels, n_gpu=n_gpu)

        output = {}
        for key, val in dst_results.items():
            if ('loss' in key) or ('acc' in key) or ('pred' in key) or ('output_bs' in key):
                output[key] = val

        policy_results = self.forward_policy_rl(dst_results, db_feat,
                                                max_words=max_words, temp=temp)

        for key, val in policy_results.items():
            output[key] = val

        return output

    def forward_policy_rl(self,dst_results,  db_feat, output_ids=None, labels=None, max_words=None, temp=0.1):
        policy_input = {}
        policy_input['hidden_utt'] = dst_results['hidden_utt']

        if not self.pretrain:
            policy_input['bs'] = self.make_bs_feature(dst_results['output_bs'])
            hidden_domain = torch.sigmoid(dst_results['hidden_domain'])
            hidden_act = torch.sigmoid(dst_results['hidden_act'])
            policy_input['act'] = torch.cat([hidden_domain, hidden_act], dim=-1)
        else:
            label_bs_ids, label_domain_ids, label_act_ids = labels
            policy_input['bs'] = self.make_bs_feature_from_label(label_bs_ids).float()
            policy_input['act'] = torch.cat([label_domain_ids, label_act_ids], dim=-1).float()

        policy_input['db'] = db_feat
        policy_input['output_ids'] = None

        policy_results = self.policy.forward_rl(policy_input, max_words, temp)

        return policy_results

    def reset_seed(self, seed=42):
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)