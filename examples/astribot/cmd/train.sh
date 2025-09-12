#!/bin/bash
# run_astribot.sh

# 遇到错误就退出
set -e

# 回到项目根目录
cd ../../../

# 定义任务名,与configs目录下的json文件名对应
task_name="wrc_pnp"

# 定义实验名,与存储模型的目录对应
exp_name="test"

#export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# 计算 norm stats
uv run scripts/compute_norm_stats_list.py --config-name $task_name

# 启动训练
uv run scripts/train_auto.py $task_name --exp-name=$exp_name --resume