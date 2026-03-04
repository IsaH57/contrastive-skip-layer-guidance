#!/bin/bash
#SBATCH --job-name=fid_sweep
#SBATCH --partition=a100       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/fid_sweep-%j.out
#SBATCH --chdir=/export/home/ru63zus/repos/msg/contrastive-skip-layer-guidance/LightningDiT
#SBATCH --gres=gpu:a100:1

pred_npz=/export/scratch/ru63zus/lightningdit/output/lightningdit_xl_vavae_f16d32-stg2-l1/lightningdit-xl-1-ckpt-lightningdit-xl-imagenet256-800ep-euler-250-interval0.11-cfg6.70-shift0.30-slg2.00.npz

python guided-diffusion/evaluations/evaluator.py \
    /export/scratch/ru63zus/assets/VIRTUAL_imagenet256_labeled.npz \
    ${pred_npz}