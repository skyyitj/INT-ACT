#!/bin/bash
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
echo "Please also set TRANSFORMERS_CACHE (Huggingface cache) and download PaliGemma weights there."

export TRANSFORMERS_CACHE="/media/jushen/linda-zhao/data1/open-pi-zero/transformer_cache"
export HF_HOME="/media/jushen/linda-zhao/data1/huggingface"

export MS2_REAL2SIM_ASSET_DIR="/media/jushen/linda-zhao/INT-ACT/third_party/ManiSkill2_real2sim/data"
export MS_ASSET_DIR="/media/jushen/linda-zhao/INT-ACT/third_party/ManiSkill/mani_skill/assets"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
source ./slurms/train_scripts/humanoid_set_path.sh
export PYTHONPATH=$PYTHONPATH:/media/jushen/linda-zhao/INT-ACT
export PYTHONPATH=$PYTHONPATH:/media/jushen/linda-zhao/INT-ACT/third_party/lerobot
# export CUDA_VISIBLE_DEVICES=0  # 注释掉以使用所有可用的 CUDA 设备

# 单机、无 IB 的常见稳定配置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # 如果没有 InfiniBand，禁用 IB
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 增大 NCCL 集体超时时间
export TORCH_NCCL_TIMEOUT=1800
# 兼容部分版本（如需要）
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800  # 30分钟
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
## CPU check
#TOTAL_CORES=$(nproc)
#echo "TOTAL_CORES=$TOTAL_CORES"
#
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"
#
## Compute OMP_NUM_THREADS (avoid division by zero)
#OMP_THREADS=$((TOTAL_CORES / NUM_GPU))
#
## Ensure OMP_NUM_THREADS is at least 1
#OMP_THREADS=$((OMP_THREADS > 0 ? OMP_THREADS : 1))
#echo "OMP_NUM_THREADS=$OMP_THREADS"
export MASTER_ADDR=127.0.0.1
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)

# Hugging Face authentication for gated repositories
# Option 1: Set HF_TOKEN environment variable (recommended for non-interactive scripts)
export HF_TOKEN="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"

# Option 2: Use huggingface-cli login (uncomment if preferred)
# huggingface-cli login --token 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

# Set PaliGemma pretrained path (can be overridden by environment variable)
# Defaults to TRANSFORMERS_CACHE/paligemma-3b-pt-224 if not set
export PALIGEMMA_PRETRAINED_PATH="${TRANSFORMERS_CACHE}/paligemma-3b-pt-224"
# export PALIGEMMA_PRETRAINED_PATH="google/paligemma-3b-pt-224"
echo "PALIGEMMA_PRETRAINED_PATH=${PALIGEMMA_PRETRAINED_PATH}"


# export NCCL_DEBUG=INFO
#--rdzv_id=$RANDOM \
#--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#--rdzv_backend=c10d \
torchrun \
--nnodes=1 \
--nproc_per_node=$NUM_GPU \
--max-restarts=0 \
--standalone \
src/agent/run.py \
--config_path config/train/pi0_baseline_bridge.yaml 

# --config_path config/train/pi0_baseline_bridge_freezevlm.yaml 





# wandb sync --entity yinuo_linda --project INT-ACT /media/jushen/linda-zhao/INT-ACT/wandb/offline-run-20251110_121008-npjs06jl

