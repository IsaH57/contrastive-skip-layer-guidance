#!/bin/bash
#SBATCH --job-name=eval_hands
#SBATCH --partition=a100       
#SBATCH --time=2:00:00         
#SBATCH --output=outputs/eval_hands-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 /export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/eval/eval_hands.py --path=/export/scratch/ru63zus/final_experiments/eccv/sd3_hands_msg_guidance_ablation_20260227_093354 --model=sd3