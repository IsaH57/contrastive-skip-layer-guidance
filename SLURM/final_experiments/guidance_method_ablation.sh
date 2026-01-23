#!/bin/bash
#SBATCH --job-name=guidance_method_ablation
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/guidance_method_ablation-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/guidance_method_ablation.py --target=counting --model=flux --output_path=/export/scratch/ru63zus/final_experiments/guidance_method_ablation --max_prompts=101
