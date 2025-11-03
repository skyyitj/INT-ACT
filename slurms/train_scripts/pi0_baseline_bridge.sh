#!/bin/bash

source ./slurms/train_scripts/set_path.sh
export PYTHONPATH=$PYTHONPATH:/home/lishuang/intact
export CUDA_VISIBLE_DEVICES=0,1

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
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
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
--config_path config/train/pi0_baseline_bridge.yaml \

