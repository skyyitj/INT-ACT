#!/bin/bash
# INT-ACT 环境变量配置文件
# 根据你的实际路径修改以下变量

# 训练数据集路径
export VLA_DATA_DIR="/home/lishuang/intact/data"

# 日志和模型保存路径
export VLA_LOG_DIR="/home/lishuang/intact/log"

# WandB 实体（如果使用）
export VLA_WANDB_ENTITY="your_wandb_entity"

# HuggingFace 缓存目录
export TRANSFORMERS_CACHE="/home/lishuang/.cache/huggingface/transformers"
export HF_HOME="/home/lishuang/.cache/huggingface"

# SIMPLER 环境变量
export MS2_REAL2SIM_ASSET_DIR="/home/lishuang/intact/third_party/ManiSkill2_real2sim/data"
export MS_ASSET_DIR="/home/lishuang/intact/third_party/ManiSkill/data"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# UV 缓存目录（可选）
export UV_CACHE_DIR="/home/lishuang/.cache/uv"
export UV_PYTHON_INSTALL_DIR="/home/lishuang/.local/share/uv/python"
