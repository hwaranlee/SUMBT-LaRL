#! /usr/bin/perl

use warnings;

#chdir "..";# working directory = ..
$data_dir='/home/hwaranlee/06_convlab/data/multiwoz/sumbt_larl_v3';
$delex_data_dir='/home/hwaranlee/06_convlab/data/multiwoz/annotation/annotated_user_da_with_span_full_patchName_convai.json';
$main_dir='convlab/modules/e2e/multiwoz/SUMBT_LaRL';
$tf_dir='tensorboard-1005v4-finetune';
$out_dir='models-1005v4-finetune';

#$ptr_larl_dir='models-1004v4-ptr-larl/task=bert-gru-sumbt-lr=1e-3-y=10-beta=0.01--simple_posterior';
#$ptr_sumbt_dir='models-1004v4-ptr-sumbt/task=bert-gru-sumbt-lr=1e-4-bs=3-hd=300-apw=';

$task_name = "bert-gru-sumbt";

@max_epochs = ("100"); 
@warmup_props = ("0.1"); 
@learning_rates = ("1e-4");
@batch_sizes = ("3");
@hidden_dims =("300");

$y_size = "10";
$beta="0.01";
$posterior='--simple_posterior';

@learning_schemes = ("");
@act_pos_weights = ("1"); #, "2", "3", "5"); 

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
foreach $learning_scheme (@learning_schemes)
{
foreach $act_pos_weight (@act_pos_weights)
{
	@params = ();
	push @params, "--data_dir $data_dir --delex_data_dir $delex_data_dir";
	push @params, "--bert_dir .pytorch_pretrained_bert";
	push @params, "--bert_model bert-base-uncased --do_lower_case";
	
	push @params, "--task_name $task_name";
	push @params, "--output_dir $main_dir/$out_dir/lr=$learning_rate-y=$y_size-beta=$beta-hd=$hidden_dim-apw=$act_pos_weight$learning_scheme";
	push @params, "--tf_dir $main_dir/$tf_dir";
	
	#push @params, "--ptr_larl_dir $main_dir/$ptr_larl_dir --ptr_sumbt_dir $main_dir/$ptr_sumbt_dir$act_pos_weight";

	push @params, "--target_slot all --nbt rnn";
	push @params, "--hidden_dim $hidden_dim";
	push @params, "--embed_size 256 --dec_cell_size 150 --beta 0.01";
	push @params, "--y_size $y_size --beta $beta $posterior";
	push @params, "$learning_scheme";

	push @params, "--num_train_epochs $max_epoch";
	push @params, "--warmup_proportion $warmup_prop";
	push @params, "--act_pos_weight $act_pos_weight";
	push @params, "--learning_rate $learning_rate";
	push @params, "--train_batch_size $batch_size";

	$param = join(" ", @params);

	$train_param = "\"--do_train --do_eval $param\"";
	#$train_param = "\"--do_eval $param\"";

	#$cmd = "sbatch slurm_exp/sumbt_larl.sh $train_param";

	$train_param = "--do_train --do_eval $param";
	$cmd = "CUDA_VISIBLE_DEVICES=4 python convlab/modules/e2e/multiwoz/SUMBT_LaRL/main.py $train_param";

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
print "Done!"
