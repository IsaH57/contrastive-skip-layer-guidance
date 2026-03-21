#!/bin/bash
#SBATCH --job-name=msg_samples
#SBATCH --partition=h200       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/generate_samples-%j.out
#SBATCH --gres=gpu:h200:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/generate_samples.py --target=ukiyo --model=flux --output_path=/export/scratch/ru63zus/final_experiments/generate_samples --num_prompts=10 --num_seeds=30 --limit_layers=3 --msg_scale=2.5
