#!/bin/bash
#SBATCH --job-name=lightningdit_fid_sweep
#SBATCH --output=logs/fid_sweep_%j.out
#SBATCH --error=logs/fid_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail

mkdir -p logs

CONFIG=configs/reproductions/lightningdit_xl_vavae_f16d32_64ep_cfg.yaml
CKPT_PATH=/path/to/your/checkpoint.pt
FID_REFERENCE=/path/to/your/VIRTUAL_imagenet256_labeled.npz

METHODS=cfg,stg,naive,msg
CFG_SCALE=6.7
SKIP_SCALE=9.0
SKIP_LAYERS=0,1,2
SKIP_WEIGHTS=1.0,1.0,1.0
FID_NUM=50000
PER_PROC_BATCH=4
EXP_NAME_PREFIX=lightningdit_xl_vavae_f16d32
OUTPUT_DIR=output

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
PRECISION=${PRECISION:-bf16}

cd LightningDiT

accelerate launch \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $NODE_RANK \
  --num_processes $(($GPUS_PER_NODE*$NNODES)) \
  --num_machines $NNODES \
  --mixed_precision $PRECISION \
  tools/fid_sweep.py \
  --config "$CONFIG" \
  --ckpt-path "$CKPT_PATH" \
  --fid-reference "$FID_REFERENCE" \
  --methods "$METHODS" \
  --cfg-scale "$CFG_SCALE" \
  --skip-scale "$SKIP_SCALE" \
  --skip-layers "$SKIP_LAYERS" \
  --skip-weights "$SKIP_WEIGHTS" \
  --fid-num "$FID_NUM" \
  --per-proc-batch-size "$PER_PROC_BATCH" \
  --exp-name-prefix "$EXP_NAME_PREFIX" \
  --output-dir "$OUTPUT_DIR" \
  --out-json "$OUTPUT_DIR/fid_sweep_results.json"
