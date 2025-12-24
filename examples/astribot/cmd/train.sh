#!/bin/bash

# Exit on error
set -e

# ------------------- Launch zsh environment -------------------
# Run subsequent commands in zsh shell
zsh <<'ZSH_EOF'

# Navigate to project root directory
cd ../../../

# Define task name (must match JSON config file name in configs directory)
task_name="wrc_pnp"

# Define experiment name (corresponds to model storage directory)
exp_name="all_data"

# Log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Log file name with exp_name, task_name, and timestamp
LOG_FILE="$LOG_DIR/${task_name}_${exp_name}_$(date +%m%d_%H%M).log"

# Environment variables
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1  # Disable Python output buffering

# Compute normalization statistics
stdbuf -oL -eL uv run scripts/compute_norm_stats_from_tasks.py --config-name $task_name 2>&1 | tee -a "$LOG_FILE"

# Start training
stdbuf -oL -eL uv run scripts/train_auto.py $task_name --exp-name=$exp_name --resume 2>&1 | tee -a "$LOG_FILE"

ZSH_EOF