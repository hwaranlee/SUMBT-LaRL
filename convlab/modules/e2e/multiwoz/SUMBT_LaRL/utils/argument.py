
import argparse

def add_parse_option(parser):
    ## Required parameters
     parser.add_argument("--data_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=True,
                         help="Bert pre-trained model selected in the list: bert-base-uncased, "
                              "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                              "bert-base-multilingual-cased, bert-base-chinese.")
     parser.add_argument("--bert_dir", default='/home/.pytorch_pretrained_bert',
                         type=str, required=False,
                         help="The directory of the pretrained BERT model")
     parser.add_argument("--task_name",
                         default=None,
                         type=str,
                         required=True,
                         help="The name of the task to train: bert-gru-sumbt, bert-lstm-sumbt"
                              "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
     parser.add_argument("--output_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
     parser.add_argument("--target_slot",
                         default='all',
                         type=str,
                         required=True,
                         help="Target slot idx to train model. ex. 'all', '0:1:2', or an excluding slot name 'attraction'" )
     parser.add_argument("--tf_dir",
                         default='tensorboard',
                         type=str,
                         required=False,
                         help="Tensorboard directory")
     parser.add_argument("--nbt",
                         default='rnn',
                         type=str,
                         required=True,
                         help="nbt type: rnn or transformer" )
     parser.add_argument("--fix_utterance_encoder",
                         action='store_true',
                         help="Do not train BERT utterance encoder")

     ## Other parameters
     parser.add_argument("--max_seq_length",
                         default=64,
                         type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
     parser.add_argument("--max_label_length",
                         default=32,
                         type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
     parser.add_argument("--max_turn_length",
                         default=22,
                         type=int,
                         help="The maximum total input turn length. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
     parser.add_argument('--hidden_dim',
                         type=int,
                         default=100,
                         help="hidden dimension used in belief tracker")
     parser.add_argument('--num_rnn_layers',
                         type=int,
                         default=1,
                         help="number of RNN layers")
     parser.add_argument('--zero_init_rnn',
                         action='store_true',
                         help="set initial hidden of rnns zero")
     parser.add_argument('--attn_head',
                         type=int,
                         default=4,
                         help="the number of heads in multi-headed attention")
     parser.add_argument("--do_train",
                         action='store_true',
                         help="Whether to run training.")
     parser.add_argument("--do_eval",
                         action='store_true',
                         help="Whether to run eval on the test set.")
     parser.add_argument("--do_analyze",
                         action='store_true',
                         help="Whether to analyze the model")
     parser.add_argument("--do_lower_case",
                         action='store_true',
                         help="Set this flag if you are using an uncased model.")
     parser.add_argument("--distance_metric",
                         type=str,
                         default="euclidean",
                         help="The metric for distance between label embeddings: cosine, euclidean.")
     parser.add_argument("--train_batch_size",
                         default=4,
                         type=int,
                         help="Total dialog batch size for training.")
     parser.add_argument("--dev_batch_size",
                         default=16,
                         type=int,
                         help="Total dialog batch size for validation.")
     parser.add_argument("--eval_batch_size",
                         default=16,
                         type=int,
                         help="Total dialog batch size for evaluation.")
     parser.add_argument("--learning_rate",
                         default=5e-5,
                         type=float,
                         help="The initial learning rate for BertAdam.")
     parser.add_argument("--lambda_domain",
                         default=10,
                         type=float,
                         help="Hyperparameter for loss_domain")
     parser.add_argument("--lambda_act",
                         default=1,
                         type=float,
                         help="Hyperparameter for loss_act")
     parser.add_argument("--domain_pos_weight",
                         default=1,
                         type=float,
                         help="Positive weight of BCE Loss")
     parser.add_argument("--act_pos_weight",
                         default=1,
                         type=float,
                         help="Positive weight of BCE Loss")
     parser.add_argument("--diff_pos_weight", type= lambda x: (str(x).lower() == 'false'), default=False)
     parser.add_argument("--num_train_epochs",
                         default=3.0,
                         type=float,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--patience",
                         default=5,
                         type=int,
                         help="The number of epochs to allow no further improvement.")
     parser.add_argument("--warmup_proportion",
                         default=0.1,
                         type=float,
                         help="Proportion of training to perform linear learning rate warmup for. "
                              "E.g., 0.1 = 10%% of training.")
     parser.add_argument("--no_cuda",
                         action='store_true',
                         help="Whether not to use CUDA when available")
     parser.add_argument("--local_rank",
                         type=int,
                         default=-1,
                         help="local_rank for distributed training on gpus")
     parser.add_argument('--seed',
                         type=int,
                         default=42,
                         help="random seed for initialization")
     parser.add_argument('--fp16',
                         action='store_true',
                         help="Whether to use 16-bit float precision instead of 32-bit")
     parser.add_argument('--loss_scale',
                         type=float, default=0,
                         help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                              "0 (default value): dynamic loss scaling.\n"
                              "Positive power of 2: static loss scaling value.\n")
     parser.add_argument("--do_not_use_tensorboard",
                         action='store_true',
                         help="Whether to run eval on the test set.")
     parser.add_argument("--pretrain_sumbt", action='store_true', help="pretrain only sumbt")
     parser.add_argument("--pretrain_larl", action='store_true', help="pretrain only larl")
     parser.add_argument("--pretrain", action='store_true', help="pretrain sumbt and larl")
     parser.add_argument("--ptr_sumbt_dir", type=str, default='', help='pretrained sumbt model dir')
     parser.add_argument("--ptr_larl_dir", type=str, default='', help='pretrained larl model dir')
     parser.add_argument("--ptr_model_dir", type=str, default='', help='pretrained sumbt+larl model dir')
     parser.add_argument("--get_utt_from_ptr", action='store_true', help="get utterance vector for policy input from ptretrained BERT")
     parser.add_argument("--eval_level", type=int, default=0, help="Task evaluation level (0, 1, 2)")

     # Policy
     parser.add_argument("--dec_max_vocab_size",
                         type=int,
                         default=1000,
                         help="Vocabulary size of decoder")
     parser.add_argument("--dec_max_seq_len",
                        type=int,
                        default=50,
                        help="Maximum decoder sequence length")

     # Categorical Latent Action
     parser.add_argument("--k_size",
                        type=int,
                        default=20,
                        help="Maximum decoder sequence length")
     parser.add_argument("--y_size",
                        type=int,
                        default=10,
                        help="Maximum decoder sequence length")
     parser.add_argument("--simple_posterior",
                         action='store_true',
                        help="Use simple posterior")
     parser.add_argument("--contextual_posterior",
                         action='store_true',
                        help="Use contextual posterior")
     parser.add_argument("--db_size", type=int, default=62)
     parser.add_argument("--embed_size", type=int, default=100)
     parser.add_argument("--utt_rnn_cell", type=str, default='gru')
     parser.add_argument("--utt_cell_size", type=int, default=300)
     parser.add_argument("--bi_utt_cell", type=lambda x: (str(x).lower() == 'true'), default=True)
     parser.add_argument("--enc_use_attn", type=lambda x: (str(x).lower() == 'true'), default=True)
     parser.add_argument("--dec_rnn_cell", type=str, default='lstm')
     parser.add_argument("--num_layers", type=int, default=1)
     parser.add_argument("--dec_cell_size", type=int, default=300)
     parser.add_argument("--dec_use_attn", type=lambda x: (str(x).lower() == 'true'), default=True)
     parser.add_argument("--dec_attn_mode", type=str, default='cat')
     parser.add_argument("--avg_type", type=str, default='word')
     parser.add_argument("--beta", type=float, default=0.001)
     parser.add_argument("--use_pr", type= lambda x: (str(x).lower() == 'true'), default=True)
     parser.add_argument("--use_mi", type= lambda x: (str(x).lower() == 'true'), default=False)
     parser.add_argument("--use_diversity", type= lambda x: (str(x).lower() == 'true'), default=False)
     parser.add_argument("--dropout", type=float, default=0.5) # used for both dst(except bert and attention) and policy
     parser.add_argument("--beam_size", type=int, default=20)
     parser.add_argument("--gen_type", type=str, default='greedy')
     parser.add_argument("--use_oracle_bs", type= lambda x: (str(x).lower() == 'true'), default=True)

     # Offpolicy reinforcement
     parser.add_argument("--do_reinforce", action='store_true', help="off-policy reinforement learning")
     parser.add_argument("--delex_data_dir", type=str, default=None, required=True, help='delexicalized data directory for evaluator')
     parser.add_argument("--record_freq", type=int, default=200)
     parser.add_argument("--sv_train_freq", type=int, default=0)
     parser.add_argument("--num_episodes", type=int, default=0)
     parser.add_argument("--tune_pi_only", type= lambda x: (str(x).lower() == 'true'), default=False)
     parser.add_argument("--rl_max_words", type=int, default=100)
     parser.add_argument("--temperature", type=float, default=1.0)
     parser.add_argument("--episode_repeat", type=float, default=1.0)
     parser.add_argument("--rl_lr", type=float, default=0.01)
     parser.add_argument("--rl_clip", type=float, default=0.5)
     parser.add_argument("--gamma", type=float, default=0.99, help='reward discount')
     parser.add_argument("--momentum", type=float, default=0.0, help='momentum for sgd')
     parser.add_argument("--nesterov", type= lambda x: (str(x).lower() == 'true'), default=False)
     
     return parser