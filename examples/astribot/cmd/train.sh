#!/bin/bash

# 遇到错误就退出
set -e


# ------------------- 启动 zsh 环境 -------------------
# 让后续命令在 zsh 中运行
zsh <<'ZSH_EOF'

# 回到项目根目录
cd ../../../


# 定义任务名,与configs目录下的json文件名对应
task_name="wrc_pnp"

# 定义实验名,与存储模型的目录对应
exp_name="all_data"

# 日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 日志文件名，包含 exp_name 和 task_name
LOG_FILE="$LOG_DIR/${task_name}_${exp_name}_$(date +%m%d_%H%M).log"




#export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1   # Python 输出不缓冲

# 计算 norm stats
stdbuf -oL -eL uv run scripts/compute_norm_stats_list.py --config-name $task_name 2>&1 | tee -a "$LOG_FILE"

# 启动训练
stdbuf -oL -eL uv run scripts/train_auto.py $task_name --exp-name=$exp_name --resume 2>&1 | tee -a "$LOG_FILE"


ZSH_EOF