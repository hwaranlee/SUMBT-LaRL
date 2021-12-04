#! /usr/bin/perl

use warnings;

#chdir "..";# working directory = ..

@target_slots = ("all"); #, "attraction", "hotel", "restaurant", "taxi", "train"); 
@task_names = ("bert-gru-sumbt"); #, "bert-gru-slot_query_multi");
@max_epochs = ("100"); #, "500","1000");
@warmup_props = ("0.1");  # ("0.003", "0.01", "0.1", "0.2", "0.5");
@learning_rates = ("1e-4"); #, "1e-4");
@batch_sizes = ("3");
@dist_metrics =("euclidean"); #"cosine
@hidden_dims =("300"); #, "600");
$tf_dir='tensorboard-multiwoz-unified-len128';
$out_dir='models-multiwoz-unified-len128';
@lambda_reqs = ("1", "2"); #, "10", "20");
@req_pos_weights = ("50", "100", "200"); #"10", "50", "100", "500");

foreach $task_name (@task_names)
{
foreach $max_epoch (@max_epochs)
{
foreach $learning_rate (@learning_rates)
{
foreach $batch_size (@batch_sizes)
{
foreach $warmup_prop (@warmup_props)
{
foreach $dist_metric (@dist_metrics)
{
foreach $hidden_dim (@hidden_dims)
{
foreach $lambda_req (@lambda_reqs)
{
foreach $req_pos_weight (@req_pos_weights)
{
#foreach $rnn_layer (@rnn_layers)
#{
#foreach $skip (@skip_connects)
#{
foreach $target_slot (@target_slots)
{
	@params = ();
	push @params, "--num_train_epochs $max_epoch";
	push @params, "--data_dir /home/hwaranlee/06_convlab/data/multiwoz/sumbt_v2.1_len128";
	push @params, "--bert_model bert-base-uncased --do_lower_case";
	push @params, "--bert_dir /home/hwaranlee/01_bert_belief_tracker/.pytorch_pretrained_bert";
	push @params, "--task_name $task_name";
	push @params, "--nbt rnn";
	push @params, "--output_dir $out_dir/task=$task_name-lr=$learning_rate-bs=$batch_size-hd=$hidden_dim-l=$lambda_req-rpw=$req_pos_weight";
	push @params, "--target_slot $target_slot";
	push @params, "--warmup_proportion $warmup_prop";
	push @params, "--lambda_req $lambda_req --lambda_gen $lambda_req --req_pos_weight $req_pos_weight";
	push @params, "--learning_rate $learning_rate";
	push @params, "--train_batch_size $batch_size";
	push @params, "--max_seq_length 128";
	push @params, "--distance_metric $dist_metric";
	push @params, "--patience 15"; #30";
	push @params, "--tf_dir $tf_dir";
	push @params, "--hidden_dim $hidden_dim"; # --num_rnn_layers $rnn_layer --skip_connect $skip";
	#push @params, "--do_not_use_tensorboard";
	#push @params, "--seed $random_seed";
	$param = join(" ", @params);

	$train_param = "\"--do_train --do_eval $param\"";
	#$train_param = "\"--do_eval $param\"";

	$cmd = "sbatch slurm_exp/nbt-multiwoz-unified.sh $train_param";
	#print "docker run -it --rm -v /gfs/nlp/hwaranlee:/home/hwaranlee -w ~/bert_belief_tracker DDP-TBRAIN-GPU01:5000/ryanne-cu9-pytorch1.0 bash \n";
	#print "docker run --rm -v /gfs/nlp/hwaranlee:/home/hwaranlee -w ~/bert_belief_tracker DDP-TBRAIN-GPU01:5000/ryanne-cu9-pytorch1.0 python3 code/Main-multislot.py --do_train  $param \n";
	#print "CUDA_VISIBLE_DEVICES=$cuda python3 code/Main-multislot.py --do_analyze $param \n";
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
}

print "Done!"



