#!/bin/bash
# 训练三个模型（§8/§9/§10）。不需要仿真，直接 python。
# 用法: bash execution_wm/scripts/train_v0.sh [model_name]
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd ~/cvpr_embed
PY=~/miniconda3/envs/isaaclab/bin/python
CFG=execution_wm/configs/train_v0.yaml
if [ -n "$1" ]; then
    $PY -m execution_wm.train.train_execution --config $CFG --model "$1"
else
    for m in action_only direct context; do
        $PY -m execution_wm.train.train_execution --config $CFG --model $m
    done
fi
