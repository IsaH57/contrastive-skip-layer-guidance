#!/bin/bash
#SBATCH --job-name=layer_count_ablation
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/layer_count_ablation-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/layer_count_ablation.py --target=complex_text_and_hands --model=flux --output_path=/export/scratch/ru63zus/final_experiments/layer_count_ablation --max_prompts=100
python3 experiment_scripts/layer_count_ablation.py --target=hands_and_aesthetics --model=flux --output_path=/export/scratch/ru63zus/final_experiments/layer_count_ablation --max_prompts=100
python3 experiment_scripts/layer_count_ablation.py --target=complex_text_and_aesthetics --model=flux --output_path=/export/scratch/ru63zus/final_experiments/layer_count_ablation --max_prompts=100
