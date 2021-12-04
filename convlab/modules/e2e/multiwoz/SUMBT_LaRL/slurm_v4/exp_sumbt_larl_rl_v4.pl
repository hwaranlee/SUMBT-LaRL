#! /usr/bin/perl

use warnings;

#chdir "..";# working directory = ..
$data_dir='/home/hwaranlee/06_convlab/data/multiwoz/sumbt_larl_v3';
$delex_data_dir='/home/hwaranlee/06_convlab/data/multiwoz/annotation/annotated_user_da_with_span_full_patchName_convai.json';
$main_dir='convlab/modules/e2e/multiwoz/SUMBT_LaRL';
$tf_dir='tensorboard-1006v4-rl-sgd';
$out_dir='models-1006v4-rl-sgd';

$ptr_model_dir='models-1005v4-finetune-e2e/lr=1e-4-y=10-beta=0.01-hd=300-apw=1';

$task_name = "bert-gru-sumbt";

@max_epochs = ("50"); # 50, 100
@warmup_props = ("0.1"); 
@learning_rates = ("1e-4"); # for sl training
@batch_sizes = ("3");
@hidden_dims =("300"); 

$y_size = "10";
$beta="0.01";
$posterior='--simple_posterior';

@act_pos_weights = ("1");
@eval_levels = ("1"); 
@sv_train_freqs = ("0"); #"1"
@use_oracle_bses = ("true"); # true

foreach $max_epoch (@max_epochs)
{
foreach $learning_rate (@learning_rates)
{
foreach $batch_size (@batch_sizes)
{
foreach $warmup_prop (@warmup_props)
{
foreach $hidden_dim (@hidden_dims)
{
foreach $act_pos_weight (@act_pos_weights)
{
foreach $sv_train_freq (@sv_train_freqs)
{
foreach $eval_level (@eval_levels)
{
foreach $use_oracle_bs (@use_oracle_bses)
{	
	@params = ();
	push @params, "--data_dir $data_dir --delex_data_dir $delex_data_dir";
	push @params, "--bert_dir .pytorch_pretrained_bert";
	push @params, "--bert_model bert-base-uncased --do_lower_case";
	
	push @params, "--task_name $task_name";
	push @params, "--output_dir $main_dir/$out_dir/lr=$learning_rate-ep=$max_epoch-apw=$act_pos_weight-lv=$eval_level-svf=$sv_train_freq-stdbias";
	push @params, "--tf_dir $main_dir/$tf_dir";

	push @params, "--ptr_model_dir $main_dir/$ptr_model_dir";

	push @params, "--target_slot all --nbt rnn";
	push @params, "--hidden_dim $hidden_dim";
	push @params, "--act_pos_weight $act_pos_weight";

	push @params, "--embed_size 256 --dec_cell_size 150 --beta 0.01";
	push @params, "--y_size $y_size --beta $beta $posterior";

	push @params, "--tune_pi_only true"; # WATCHOUT!
	push @params, "--eval_level $eval_level";
	#push @params, "--eval_level 0";
	push @params, "--sv_train_freq $sv_train_freq";
	push @params, "--use_oracle_bs $use_oracle_bs";
	push @params, "--patience 30";

	push @params, "--num_train_epochs $max_epoch";
	push @params, "--warmup_proportion $warmup_prop";
	push @params, "--learning_rate $learning_rate";
	push @params, "--train_batch_size $batch_size";

	$param = join(" ", @params);

	$train_param = "\"--do_reinforce $param\"";
	#$train_param = "\"--do_eval $param\"";

	#$cmd = "sbatch slurm_exp/sumbt_larl.sh $train_param";

	$train_param = "--do_reinforce $param";
	$cmd = "CUDA_VISIBLE_DEVICES=5 python convlab/modules/e2e/multiwoz/SUMBT_LaRL/main.py $train_param";


print "$cmd\n";
system $cmd;
sleep 1;
}
}
}
}
}
}
}
}
}
print "Done!"
