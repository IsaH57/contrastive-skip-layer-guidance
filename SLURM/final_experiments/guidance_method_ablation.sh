#!/bin/bash
#SBATCH --job-name=guidance_method_ablation
#SBATCH --partition=h200       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/guidance_method_ablation-%j.out
#SBATCH --gres=gpu:h200:1

cd ..
cd ..
nvidia-smi
python3 /export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/experiment_scripts/guidance_method_ablation.py --target=ukiyo --model=sd35 --output_path=/export/scratch/ru63zus/final_experiments/eccv --max_prompts=101
