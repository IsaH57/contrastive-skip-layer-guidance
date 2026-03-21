#!/bin/bash
#SBATCH --job-name=eval_coco
#SBATCH --partition=a100     
#SBATCH --time=2:00:00         
#SBATCH --output=outputs/eval_coco-%j.out
#SBATCH --gres=gpu:a100:1

cd ..
cd ..
nvidia-smi
python3 /export/scratch/ru63zus/repos/contrastive-skip-layer-guidance/COCO_FID_stuff/eval_coco_pregenerated.py \
    --coco-dir /export/scratch/ra97ram/datasets/coco2017val \
    --generated-dir final_experiments/layer_ablations/flux_coco_cfgskip_False_layer_ablation_20260226_191458 \
    --limit 1024