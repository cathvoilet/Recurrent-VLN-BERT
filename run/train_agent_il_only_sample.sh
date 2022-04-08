#!/bin/bash --login
#SBATCH --job-name=train_vlnbert_agent_il-rs116
#SBATCH --output=slurm_logs/train_vlnbert_agent_il-rs116.out
#SBATCH --error=slurm_logs/train_vlnbert_agent_il-rs116.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --mem=32gb
#SBATCH --gres=gpu:p6000

set -x
module add cuda/8.0.44 cudnn/v5.1

DOCKER=/vulcanscratch/lzhao/docker/vlnbert.sif

name=VLNBERT-train-il-rs116

flag="--vlnbert prevalent
      --test_only 0
      --train listener
      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 5e-6
      --iters 300000
      --seed 116
      --train_sampling 0.9
      --no_rl 1
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name

singularity exec --bind /vulcanscratch:/vulcanscratch --nv $DOCKER python3 -u r2r_src/train.py $flag --name $name

