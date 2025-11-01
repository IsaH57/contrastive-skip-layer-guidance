#!/bin/bash
#SBATCH --job-name=layer_ablation
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/layer_ablation-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/MSG_guidance_sweep.py --target=aesthetics --model=sd3 --output_path=/export/scratch/ru63zus/final_experiments/layer_ablations --max_prompts=1

