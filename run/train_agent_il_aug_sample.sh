#!/bin/bash --login
#SBATCH --job-name=train_vlnbert_il_aug_rs120
#SBATCH --output=slurm_logs/train_vlnbert_il_aug_rs120.out
#SBATCH --error=slurm_logs/train_vlnbert_il_aug_rs120.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=tron
#SBATCH --qos=default
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000

set -x
module add cuda/9.1.85 cudnn/v7.5.0

DOCKER=/fs/nexus-scratch/lzhao/docker/vlnbert.sif

name=VLNBERT-train-il-aug-rs120

flag="--vlnbert prevalent
      --aug /fs/nexus-scratch/lzhao/repos/Matterport3DSimulator/data/prevalent/prevalent_aug.json
      --test_only 0
      --train auglistener
      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 5e-6
      --iters 300000
      --seed 120
      --train_sampling 0.9
      --no_rl 1
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name

singularity exec --bind /fs/nexus-scratch:/fs/nexus-scratch --nv $DOCKER python3 -u r2r_src/train.py $flag --name $name

