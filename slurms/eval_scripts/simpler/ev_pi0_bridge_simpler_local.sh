#!/bin/bash

# 本地服务器版本的评估脚本
# 已移除SLURM和Singularity容器相关配置，适配本地环境
# 使用conda环境管理（需要手动激活）

# ========================================
# 使用前请先手动激活conda环境：
# conda activate your_env_name
# ========================================

# 配置文件和随机种子
CONFIG_NAMES=("pi0_finetune_bridge_ev.yaml")
SEEDS=(42 7 314)

# 检查端口是否被占用的函数（保留但未使用）
is_port_in_use() {
    ss -tuln | grep ":$1" > /dev/null
    return $?
}

# 查找可用端口的函数
find_available_port() {
    local port
    for port in $(shuf -i 10000-65500 -n 200); do
        if ! ss -tuln | grep ":$port" > /dev/null; then
            echo $port
            return 0
        fi
    done
    # 如果没有找到随机端口，返回默认端口（不太可能发生）
    echo 5000
    return 1
}

# 显示GPU信息
echo "检测到的GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "当前使用的conda环境: $CONDA_DEFAULT_ENV"

# 遍历随机种子
for SEED in "${SEEDS[@]}"; do
    echo "============================================"
    echo "正在使用随机种子: $SEED"
    echo "============================================"

    # 遍历配置文件
    for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
        echo "正在使用配置文件: $CONFIG_NAME"
        
        # 使用Python读取配置文件中的gradient step列表
        STEP_COUNTS=( $(python3 - <<EOF
import yaml, os
from yaml.loader import SafeLoader

# 处理!include指令，加载引用的文件
def include_constructor(loader, node):
    base = os.path.dirname(os.path.abspath("config/experiment/simpler/${CONFIG_NAME}"))
    rel = loader.construct_scalar(node)
    with open(os.path.join(base, rel)) as f:
        return yaml.load(f, Loader=SafeLoader)

SafeLoader.add_constructor('!include', include_constructor)

cfg_path = "config/experiment/simpler/${CONFIG_NAME}"
with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=SafeLoader)

for s in cfg["eval_cfg"]["pretrained_model_gradient_step_cnt"]:
    print(s)
EOF
) )
        
        echo "将评估以下checkpoint步数: ${STEP_COUNTS[@]}"
        
        # 批量大小：同时运行的服务器-客户端对数量
        # 根据您的GPU显存调整此值（80GB建议4个，更小的GPU可以减少）
        BATCH_SIZE=4
        TOTAL=${#STEP_COUNTS[@]}
        
        # 分批评估
        for (( i=0; i<TOTAL; i+=BATCH_SIZE )); do
            CHUNK=( "${STEP_COUNTS[@]:i:BATCH_SIZE}" )
            echo "--------------------------------------------"
            echo "正在评估批次: ${CHUNK[@]}"
            echo "--------------------------------------------"
            
            # 为每个step启动一个服务器+客户端对，每对使用独立端口
            SERVER_PIDS=()
            CLIENT_PIDS=()
            
            for STEP in "${CHUNK[@]}"; do
                # 查找可用端口
                PORT=$(find_available_port)
                echo "为step $STEP 选择端口: $PORT"

                echo "在端口 $PORT 上启动服务器 (step $STEP)..."
                
                # 启动服务器
                python src/agent/run.py \
                    --config_path config/experiment/simpler/${CONFIG_NAME} \
                    --seed ${SEED} \
                    --use_bf16 False \
                    --eval_cfg.port ${PORT} \
                    --eval_cfg.pretrained_model_gradient_step_cnt="[${STEP}]" \
                    --use_wandb False \
                    --eval_cfg.role server &
                SERVER_PIDS+=($!)
                
                echo "在端口 $PORT 上启动客户端 (step $STEP)..."
                # 给服务器一点时间绑定端口
                sleep 2
                
                # 启动客户端
                # 注意：如果服务器和客户端需要不同的环境，需要修改此处
                # 例如使用: conda run -n simpler_env python src/agent/run.py ...
                python src/agent/run.py \
                    --config_path config/experiment/simpler/${CONFIG_NAME} \
                    --seed ${SEED} \
                    --use_bf16 False \
                    --eval_cfg.port ${PORT} \
                    --eval_cfg.pretrained_model_gradient_step_cnt="[${STEP}]" \
                    --use_wandb False \
                    --eval_cfg.role client &
                CLIENT_PIDS+=($!)
                
            done
            
            # 等待当前批次的所有客户端完成
            echo "等待当前批次的客户端完成..."
            wait "${CLIENT_PIDS[@]}"
            
            # 然后终止当前批次的所有服务器
            echo "终止当前批次的服务器..."
            for pid in "${SERVER_PIDS[@]}"; do 
                kill $pid 2>/dev/null
            done
            
            echo "批次 ${CHUNK[@]} 评估完成"
        done
        
        echo "配置文件 $CONFIG_NAME 评估完成"
    done
    
    echo "随机种子 $SEED 评估完成"
done

echo "============================================"
echo "所有评估任务完成！"
echo "============================================"

