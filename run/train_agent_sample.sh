#!/bin/bash --login
#SBATCH --job-name=train_vlnbert_rs123
#SBATCH --output=slurm_logs/train_vlnbert_rs123.out
#SBATCH --error=slurm_logs/train_vlnbert_rs123.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --gres=gpu:p6000

set -x
module add cuda/8.0.44 cudnn/v5.1

DOCKER=/vulcanscratch/lzhao/docker/vlnbert.sif

name=VLNBERT-train-rs123

flag="--vlnbert prevalent
      --aug /vulcanscratch/lzhao/repos/Matterport3DSimulator/data/prevalent/prevalent_aug.json
      --test_only 0
      --train auglistener
      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 1e-5
      --iters 300000
      --seed 123
      --train_sampling 0.9
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name

singularity exec --bind /vulcanscratch:/vulcanscratch --nv $DOCKER python3 -u r2r_src/train.py $flag --name $name

