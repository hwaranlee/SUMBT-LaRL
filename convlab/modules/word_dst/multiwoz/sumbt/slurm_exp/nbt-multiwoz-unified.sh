#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
###SBATCH --exclude=DDP-TBRAIN-GPU16
###SBATCH --partition=P40

echo $SLURM_JOB_GPUS

working_dir='/home/hwaranlee/06_convlab/convlab/modules/word_dst/multiwoz/sumbt'

srun docker run --rm -v /gfs/nlp/hwaranlee:/home/hwaranlee -w $working_dir -e CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS DDP-TBRAIN-GPU01:5000/ryanne-cu9-pytorch1.0:2.4 python3 code/main.py $1
