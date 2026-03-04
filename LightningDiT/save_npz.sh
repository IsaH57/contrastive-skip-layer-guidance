#!/bin/bash
#SBATCH --job-name=fid_sweep
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/fid_sweep-%j.out
#SBATCH --chdir=/export/home/ru63zus/repos/msg/contrastive-skip-layer-guidance/LightningDiT
#SBATCH --gres=gpu:a100:1

python3 /export/home/ru63zus/repos/msg/contrastive-skip-layer-guidance/LightningDiT/tools/save_npz.py