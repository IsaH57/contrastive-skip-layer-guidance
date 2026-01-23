#!/bin/bash
#SBATCH --job-name=msg_samples
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/generate_samples-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/generate_samples.py --target=hands --model=sd35 --output_path=/export/scratch/ru63zus/final_experiments/generate_samples --num_prompts=20 --num_seeds=25 --limit_layers=2 --msg_scale=2.0
python3 experiment_scripts/generate_samples.py --target=aesthetics --model=sd35 --output_path=/export/scratch/ru63zus/final_experiments/generate_samples --num_prompts=20 --num_seeds=25 --limit_layers=2 --msg_scale=2.0
python3 experiment_scripts/generate_samples.py --target=text --model=sd35 --output_path=/export/scratch/ru63zus/final_experiments/generate_samples --num_prompts=20 --num_seeds=25 --limit_layers=2 --msg_scale=2.0

