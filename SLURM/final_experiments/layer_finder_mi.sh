#!/bin/bash
#SBATCH --job-name=MI_finder
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/find_layers-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 experiment_scripts/layer_finding_MI.py --target=aesthetics --model=sd3 --output_path=contrastive-skip-layer-guidance/experiment_scripts/final_results