#!/bin/bash
#SBATCH --job-name=pi_vote_debug
#SBATCH --output=slurm_logs/pi_vote_debug.out
#SBATCH --error=slurm_logs/pi_vote_debug.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1

set -x
module add cuda/8.0.44 cudnn/v5.1

DOCKER=/vulcanscratch/lzhao/docker/vlnbert.sif


exp_dir="experiments/agents_vote_debug/"
mkdir -p $exp_dir

voting_agents=("VLNBERT-train-Prevalent" "VLNBERT-train-rs123")

for agent in "${voting_agents[@]}"; do
    flag="--vlnbert prevalent
      --submit 1
      --test_only 0
      --train eval_listener_outputs
      --speaker_output_files /vulcanscratch/lzhao/repos/Matterport3DSimulator/speaker_outputs/speaker11_val_seen_eval.pred /vulcanscratch/lzhao/repos/Matterport3DSimulator/speaker_outputs/speaker11_val_unseen_eval.pred
      --load snap/${agent}/state_dict/best_val_unseen
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
    name="${agent}_pi_vote"
    mkdir -p snap/$name

    singularity exec --bind /vulcanscratch:/vulcanscratch --nv $DOCKER python3 -u r2r_src/train.py $flag --name $name
done

python3 prag_inf/vote_instructions.py --output_exp "$exp_dir" -input_exps "${voting_agents[@]}"


testing_agents=("VLNBERT-PREVALENT-final")

for agent in "${testing_agents[@]}"; do
    flag="--vlnbert prevalent
      --submit 1
      --test_only 0
      --train eval_listener_outputs
      --speaker_output_files ${exp_dir}/voted_best_avg_val_seen_eval.json ${exp_dir}/voted_best_avg_val_unseen_eval.json
      --load snap/${agent}/state_dict/best_val_unseen
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
    name="${agent}_pi_test"
    mkdir -p snap/$name

    singularity exec --bind /vulcanscratch:/vulcanscratch --nv $DOCKER python3 -u r2r_src/train.py $flag --name $name
done

