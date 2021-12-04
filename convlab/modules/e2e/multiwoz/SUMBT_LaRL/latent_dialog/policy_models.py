import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.base_models import BaseModel
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.utils.processor import EOS, PAD, BOS
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog import nn_lib
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.utils import INT, FLOAT, LONG, Pack, cast_type
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.enc2dec.decoders import DecoderRNN, GEN, TEACH_FORCE
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.enc2dec.encoders import RnnUttEncoder
from convlab.modules.e2e.multiwoz.SUMBT_LaRL.latent_dialog.criterions import NLLEntropy, CatKLLoss, Entropy, NormKLLoss

import pdb

class SysPerfectBD2Cat(BaseModel):
    def __init__(self, config, processor, device):
        self.device = device
        self.use_gpu = True if self.device.type == 'cuda' else False
        config.use_gpu = self.use_gpu

        super(SysPerfectBD2Cat, self).__init__(config)
        self.vocab = processor.vocab
        self.vocab_dict = processor.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = processor.bs_size * 2
        self.db_size = processor.db_size
        self.act_size = len(processor.ontology_domain) + len(processor.ontology_act)
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior
        self.utt_emb_size = config.utt_emb_size
        self.embedding = None
        
        if not self.simple_posterior:
            self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                            embedding_dim=config.embed_size,
                                            feat_size=0,
                                            goal_nhid=0,
                                            rnn_cell=config.utt_rnn_cell,
                                            utt_cell_size=config.utt_cell_size,
                                            num_layers=config.num_layers,
                                            input_dropout_p=config.dropout,
                                            output_dropout_p=config.dropout,
                                            bidirectional=config.bi_utt_cell,
                                            variable_lengths=False,
                                            use_attn=config.enc_use_attn,
                                            embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Discrete(self.utt_emb_size + self.db_size + self.bs_size + self.act_size, 
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_emb_size + self.utt_encoder.output_size + self.db_size + self.bs_size + self.act_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.dec_max_seq_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = th.log(th.ones(1) / config.k_size).to(self.device)
        self.eye = th.eye(self.config.y_size).unsqueeze(0).to(self.device)
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):

        utt_summary = data_feed['hidden_utt'] # [ds*ts, hidden_dim] #ds = num dialog, ts = num turns
        batch_size = utt_summary.size(0)

        bs_label = data_feed['bs'].view(batch_size, -1)
        act_label = data_feed['act'].view(batch_size, -1)
        db_label = data_feed['db'].float().view(batch_size, -1)
        
        if mode == GEN:
            out_utts = None
            dec_inputs = None
            labels = None
        else:
            out_utts = data_feed['output_ids'].view(batch_size, -1)

            # get decoder inputs
            dec_inputs = out_utts[:, :-1]
            labels = out_utts[:, 1:].contiguous()
            # get padding utterances
            valid_turn = (out_utts != self.pad_id).sum(-1).nonzero().squeeze()
        
        # create decoder initial states
        enc_last = th.cat([bs_label, act_label, db_label, utt_summary], dim=1)

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=True)
                log_qy = log_py
            else:
                # encode response and use posterior to find q(z|x, c)
                x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
                if self.contextual_posterior:
                    logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
                else:
                    logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

                sample_y = self.gumbel_connector(logits_qy, hard=False)
            
        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict

        else:
            result = Pack()
            # regularization qy to be uniform
            # select valid log_qy
            log_qy = log_qy.view(-1, self.config.y_size, self.config.k_size)
            log_qy = log_qy.index_select(0, valid_turn)
            
            if not self.simple_posterior:
                # select valid log_py
                log_py = log_py.view(-1, self.config.y_size, self.config.k_size)
                log_py = log_py.index_select(0, valid_turn)

            batch_size = len(valid_turn)

            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)

            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)

            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        
        utt_summary = data_feed['hidden_utt'] # [ds*ts, hidden_dim] #ds = num dialog, ts = num turns
        batch_size = utt_summary.size(0)      # batch_size = ds * ts

        bs_label = data_feed['bs'].view(batch_size, -1)
        act_label = data_feed['act'].view(batch_size, -1)
        db_label = data_feed['db'].float().view(batch_size, -1)
        
        # context input
        enc_last = th.cat([bs_label, act_label, db_label, utt_summary], dim=1)
        
        # calculate latent distribution
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)
        
        # latent -> reinforce : joint_logpz, sample_y
        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        # sample_y = cast_type(th.zeros(log_qy.size()), FLOAT, self.use_gpu)
        sample_y = th.zeros(log_qy.size()).to(self.device)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        
        result = {}
        result['logprobs'] = logprobs
        result['outs'] = outs
        result['joint_logpz'] = joint_logpz
        result['sample_y'] = sample_y
        
        return result
