#!/bin/bash
#SBATCH --job-name=combined_concepts
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/combined_concepts-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/combined_concepts.py --target=complex_text_hands --model=sd3 --output_path=/export/scratch/ru63zus/final_experiments/combined_concepts --max_prompts=100
