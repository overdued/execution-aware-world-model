#!/bin/bash
# V0 数据采集（first_work.md Step 2-4）
# 用法: bash execution_wm/scripts/collect_v0.sh [--conditions normal] [--num-episodes 10]
set -e
cd ~/IsaacLab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
    ~/cvpr_embed/execution_wm/data/collector.py \
    --config ~/cvpr_embed/execution_wm/configs/collect_v0.yaml "$@"
