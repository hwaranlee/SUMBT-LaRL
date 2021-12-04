#! /bin/bash

data_dir='/home/hwaranlee/06_convlab/data/multiwoz/sumbt_larl'
delex_data_dir='/home/hwaranlee/06_convlab/data/multiwoz/annotation/annotated_user_da_with_span_full_patchName_convai.json'
main_dir='convlab/modules/e2e/multiwoz/SUMBT_LaRL'

#tf_dir='tensorboard-1005v4-finetune-e2e'
#out_dir='models-1005v4-finetune-e2e'
#ptr_model_dir='models-1005v4-finetune-e2e/lr=1e-4-y=10-beta=0.01-hd=300-apw=1'

tf_dir='tensorboard-1005v4-finetune-before'
out_dir='models-1005v4-finetune-before'
ptr_model_dir='models-1005v4-finetune-before/lr=1e-4-y=10-beta=0.01-hd=300-apw=1'

task_name="bert-gru-sumbt"

max_epoch="100"
warmup_prop="0.1" 
learning_rate="1e-4" # for sl training
batch_size="3"
hidden_dim="300" 

y_size="10" 
beta="0.01"
posterior='--simple_posterior'

act_pos_weight="1"

test_level="0"

gpu=0

OPTIND=1
while getopts "t:g:" opt; do
        case "$opt" in
                t) test_level=$OPTARG;;
                g) gpu=$OPTARG;;
        esac
done
shift $((OPTIND -1))


params="--data_dir $data_dir --delex_data_dir $delex_data_dir"
params=$params" --bert_dir .pytorch_pretrained_bert"
params=$params" --bert_model bert-base-uncased --do_lower_case"

params=$params" --task_name $task_name"
params=$params" --output_dir $main_dir/$out_dir/lr=$learning_rate-y=$y_size-beta=$beta-hd=$hidden_dim-apw=$act_pos_weight"
params=$params" --tf_dir $main_dir/$tf_dir"

params=$params" --ptr_model_dir $main_dir/$ptr_model_dir"

params=$params" --target_slot all --nbt rnn"
params=$params" --hidden_dim $hidden_dim"
params=$params" --embed_size 256 --dec_cell_size 150 --beta 0.01"
params=$params" --y_size $y_size --beta $beta $posterior"

params=$params" --num_train_epochs $max_epoch"
params=$params" --warmup_proportion $warmup_prop"
params=$params" --act_pos_weight $act_pos_weight"
params=$params" --learning_rate $learning_rate"
params=$params" --train_batch_size $batch_size"

params=$params" --eval_level $test_level"
train_param="--do_eval $params"
cmd="CUDA_VISIBLE_DEVICES=$gpu python convlab/modules/e2e/multiwoz/SUMBT_LaRL/main.py $train_param"

echo $cmd

CUDA_VISIBLE_DEVICES=$gpu python convlab/modules/e2e/multiwoz/SUMBT_LaRL/main.py $train_param
