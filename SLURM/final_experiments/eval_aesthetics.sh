#!/bin/bash
#SBATCH --job-name=eval_aesthetics
#SBATCH --partition=a100       
#SBATCH --time=2:00:00         
#SBATCH --output=outputs/eval_aesthetics-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 eval/eval_aesthetics.py --path=/export/scratch/ru63zus/final_experiments/guidance_method_ablation/flux_aesthetics_msg_guidance_ablation_20251104_174600 --model=flux