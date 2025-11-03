#!/bin/bash

# 设置错误时立即退出
set -e

# 如果环境变量未设置，使用默认值（避免交互式输入卡住）
if [ -z "$VLA_DATA_DIR" ]; then
    export VLA_DATA_DIR="${VLA_DATA_DIR:-/data1/open-pi-zero/data}"
fi
if [ -z "$VLA_LOG_DIR" ]; then
    export VLA_LOG_DIR="${VLA_LOG_DIR:-/data1/open-pi-zero/log}"
fi
if [ -z "$TRANSFORMERS_CACHE" ]; then
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data1/open-pi-zero/transformer_cache}"
fi
if [ -z "$HF_HOME" ]; then
    export HF_HOME="${HF_HOME:-/data1/huggingface}"
fi

export MS2_REAL2SIM_ASSET_DIR="/home/lishuang/intact/third_party/ManiSkill2_real2sim/mani_skill2_real2sim/assets"
export MS_ASSET_DIR="/home/lishuang/intact/third_party/ManiSkill/mani_skill/assets"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Using VLA_DATA_DIR=$VLA_DATA_DIR"
echo "Using VLA_LOG_DIR=$VLA_LOG_DIR"

export PYTHONPATH=$PYTHONPATH:/home/lishuang/intact
export CUDA_VISIBLE_DEVICES=0,1

# # 单机、无 IB 的常见稳定配置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           # 如果没有 InfiniBand，禁用 IB
export NCCL_P2P_DISABLE=1          # 禁用P2P通信，改用shared memory
export NCCL_SHM_DISABLE=0          # 启用共享内存通信
export CUDA_DEVICE_MAX_CONNECTIONS=1

# # 单机、无 IB 的常见稳定配置
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1           # 如果没有 InfiniBand，禁用 IB
# export NCCL_P2P_DISABLE=0
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# 增大 NCCL 集体超时时间
export TORCH_NCCL_TIMEOUT=1800
# 兼容部分版本（如需要）
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800  # 30分钟
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0

# 添加更多调试信息
export NCCL_DEBUG_SUBSYS=ALL
## CPU check
#TOTAL_CORES=$(nproc)
#echo "TOTAL_CORES=$TOTAL_CORES"
#
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

# 验证 GPU 可用性
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export MASTER_ADDR=127.0.0.1
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)
echo "Using MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

echo "Starting torchrun with $NUM_GPU GPUs..."
echo "Command: torchrun --nnodes=1 --nproc_per_node=$NUM_GPU --max-restarts=0 --standalone src/agent/run.py --config_path config/train/pi0_baseline_bridge.yaml"
echo "=========================================="

torchrun \
--nnodes=1 \
--nproc_per_node=$NUM_GPU \
--max-restarts=0 \
--standalone \
src/agent/run.py \
--config_path config/train/pi0_baseline_bridge_freezevlm.yaml
# --config_path config/train/pi0_baseline_bridge.yaml

