#!/bin/bash
# 评估 + 出图（§15/§16/§18）
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd ~/cvpr_embed
PY=~/miniconda3/envs/isaaclab/bin/python
CFG=execution_wm/configs/train_v0.yaml
$PY -m execution_wm.eval.evaluate_execution --config $CFG
$PY -m execution_wm.eval.visualize_execution --config $CFG --model context
