#!/bin/bash --login
#SBATCH --job-name=test_agent_il_pi
#SBATCH --output=slurm_logs/test_agent_il_pi.out
#SBATCH --error=slurm_logs/test_agent_il_pi.err
#SBATCH --time=3-12:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --gres=gpu:p6000

set -x
module add cuda/8.0.44 cudnn/v5.1

DOCKER=/vulcanscratch/lzhao/docker/vlnbert.sif

name=VLNBERT-test-il-pi

flag="--vlnbert prevalent
      --submit 1
      --test_only 0
      --train eval_listener_outputs
      --speaker_output_files /vulcanscratch/lzhao/repos/Matterport3DSimulator/speaker_outputs/original_val_seen_eval.pred /vulcanscratch/lzhao/repos/Matterport3DSimulator/speaker_outputs/original_val_unseen_eval.pred
      --load snap/VLNBERT-train-Prevalent/state_dict/best_val_unseen
      --features places365
      --maxAction 15
      --batchSize 8
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name

singularity exec --bind /vulcanscratch:/vulcanscratch --nv $DOCKER python3 -u r2r_src/train.py $flag --name $name

