#!/bin/bash

target_slot='all'
bert_dir='/home/hwaranlee/06_convlab/.pytorch_pretrained_bert'
cuda=5


data_dir='/home/hwaranlee/06_convlab/data/multiwoz/sumbt_larl_v3'
delex_data_dir='/home/hwaranlee/06_convlab/data/multiwoz/annotation/annotated_user_da_with_span_full_patchName_convai.json'
main_dir='convlab/modules/e2e/multiwoz/SUMBT_LaRL'
#out_dir='models-1004v4-ptr-larl'
#ptr_larl_dir=$main_dir/$out_dir/'task=bert-gru-sumbt-lr=1e-3-y=10-beta=0.01--simple_posterior'
#ptr_sumbt_dir=$main_dir/'models-1004v4-ptr-sumbt/task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-apw=5'
#ptr_model_dir=$main_dir/'models-1005v4-finetune/lr=1e-4-y=10-beta=0.01-hd=300-apw=1'
ptr_model_dir=$main_dir/'sumbtlarl_4'

output_dir=$main_dir/exp-multiwoz/exp-5

CUDA_VISIBLE_DEVICES=$cuda python $main_dir/main.py --do_analyze --num_train_epochs 10 \
	--data_dir $data_dir --bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir \
	--task_name bert-gru-sumbt --nbt rnn --output_dir $output_dir --target_slot all \
	--warmup_proportion 0.1 --learning_rate 1e-4 --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
	--patience 15 --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 \
	--embed_size 256 --dec_cell_size 150 --beta 0.01 \
	--delex_data_dir $delex_data_dir \
	--simple_posterior \
	--ptr_model_dir $ptr_model_dir \
	--tune_pi_only true --eval_level 0



#	--ptr_larl_dir $ptr_larl_dir --ptr_sumbt_dir $ptr_sumbt_dir \
	
	
	