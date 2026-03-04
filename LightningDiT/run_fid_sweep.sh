#!/bin/bash
#SBATCH --job-name=fid_sweep
#SBATCH --partition=h200       
#SBATCH --time=24:00:00
#SBATCH --output=outputs/fid_sweep-%j.out
#SBATCH --chdir=/export/home/ru63zus/repos/msg/contrastive-skip-layer-guidance/LightningDiT
#SBATCH --gres=gpu:h200:1

set -euo pipefail

REPO_DIR=${REPO_DIR:-/export/home/ru63zus/repos/msg/contrastive-skip-layer-guidance/LightningDiT}
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

LOG_DIR=${LOG_DIR:-/export/scratch/ru63zus/lightningdit/logs}
OUTPUT_DIR=${OUTPUT_DIR:-/export/scratch/ru63zus/lightningdit/output}
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

CONFIG="$REPO_DIR/configs/lightningdit_xl_vavae_f16d32.yaml"
CKPT_PATH=/export/scratch/ru63zus/ckpts/lightningdit-xl-imagenet256-800ep.pt
FID_REFERENCE=/export/scratch/ru63zus/assets/VIRTUAL_imagenet256_labeled.npz
METHODS=cfg
CFG_SCALE=6.7
SKIP_SCALE=2.0
SKIP_LAYERS=1,24,22
SKIP_WEIGHTS=1.0,0.6,0.2
FID_NUM=50000
PER_PROC_BATCH=64
NUM_SAMPLING_STEPS=250
EXP_NAME_PREFIX=lightningdit_xl_vavae_f16d32
OUTPUT_DIR=${OUTPUT_DIR}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
PRECISION=${PRECISION:-bf16}

accelerate launch \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $NODE_RANK \
  --num_processes $(($GPUS_PER_NODE*$NNODES)) \
  --num_machines $NNODES \
  --mixed_precision $PRECISION \
  "$REPO_DIR/tools/fid_sweep.py" \
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
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --exp-name-prefix "$EXP_NAME_PREFIX" \
  --output-dir "$OUTPUT_DIR" \
  --out-json "$OUTPUT_DIR/fid_sweep_results.json"
