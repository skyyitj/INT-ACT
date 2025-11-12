#!/bin/bash

#SBATCH --job-name=ev_pi0_bridge_simpler
#SBATCH --output=log/slurm/eval/simpler/%x_%j.out
#SBATCH --error=log/slurm/eval/simpler/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="a100|h100"
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --account=pr_109_tandon_advanced
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
# Exclude localhost and local addresses from proxy for websockets communication
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1
echo "Please also set TRANSFORMERS_CACHE (Huggingface cache) and download PaliGemma weights there."

export TRANSFORMERS_CACHE="/media/jushen/linda-zhao/data1/open-pi-zero/transformer_cache/"
# export HF_HOME="/media/jushen/linda-zhao/data1/huggingface"

export MS2_REAL2SIM_ASSET_DIR="/media/jushen/linda-zhao/INT-ACT/third_party/ManiSkill2_real2sim/data"
export MS_ASSET_DIR="/media/jushen/linda-zhao/INT-ACT/third_party/ManiSkill/mani_skill/assets"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
source ./slurms/train_scripts/humanoid_set_path.sh
export PYTHONPATH=$PYTHONPATH:/media/jushen/linda-zhao/INT-ACT
export PYTHONPATH=$PYTHONPATH:/media/jushen/linda-zhao/INT-ACT/third_party/lerobot
export CUDA_VISIBLE_DEVICES=0

export HF_TOKEN="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"
export PALIGEMMA_PRETRAINED_PATH="${TRANSFORMERS_CACHE}/paligemma-3b-pt-224"
# export PALIGEMMA_PRETRAINED_PATH="google/paligemma-3b-pt-224"
echo "PALIGEMMA_PRETRAINED_PATH=${PALIGEMMA_PRETRAINED_PATH}"

# # Trap Ctrl+C and clean up all child processes
# trap "echo 'Ctrl+C received, killing server...'; kill $SERVER_PID; exit 1" SIGINT

CONFIG_NAMES=("pi0_finetune_bridge_ev.yaml")
# SEEDS=(42 7 314)
SEEDS=(42)
# Function to check if a port is available
is_port_in_use() {
    ss -tuln | grep ":$1" > /dev/null
    return $?
}

find_available_port() {
    local port
    for port in $(shuf -i 10000-65500 -n 200); do
        if ! ss -tuln | grep ":$port" > /dev/null; then
            echo $port
            return 0
        fi
    done
    # Fallback to a default port if no random port is found (unlikely)
    echo 5000
    return 1
}

# # set all the paths to environment variables
# source ./set_path.sh

for SEED in "${SEEDS[@]}"; do
    echo "Running with seed $SEED"

    for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
        echo "Running with config $CONFIG_NAME"
        # pull the pretrained_model_gradient_step_cnt list via Python+PyYAML with a custom !include handler
        STEP_COUNTS=( $(python3 - <<EOF
import yaml, os
from yaml.loader import SafeLoader

# handle !include by loading the referenced file
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
        BATCH_SIZE=1
        TOTAL=${#STEP_COUNTS[@]}
        for (( i=0; i<TOTAL; i+=BATCH_SIZE )); do
            CHUNK=( "${STEP_COUNTS[@]:i:BATCH_SIZE}" )
            # spawn one server+client pair per step count, each on its own port
            SERVER_PIDS=()
            CLIENT_PIDS=()
            for STEP in "${CHUNK[@]}"; do
                # Find a random available port instead of incrementing
                PORT=$(find_available_port)
                echo "Selected random port $PORT for step $STEP"

                echo "Launching server on port $PORT for step $STEP"

                # start server for this STEP (local execution without singularity)
                python src/agent/run.py \
                    --config_path config/experiment/simpler/${CONFIG_NAME} \
                    --seed ${SEED} \
                    --use_bf16 False \
                    --eval_cfg.port ${PORT} \
                    --eval_cfg.pretrained_model_gradient_step_cnt="[${STEP}]" \
                    --use_wandb False \
                    --eval_cfg.role server &
                SERVER_PIDS+=($!)

                echo "Launching client on port $PORT for step $STEP"
                # give server a moment to bind
                sleep 2

                # start client for this STEP (local execution without singularity)
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

            # wait for this batch of clients
            wait "${CLIENT_PIDS[@]}"
            # then kill this batch of servers
            for pid in "${SERVER_PIDS[@]}"; do kill $pid; done
        done

        # # Run the gather_data_to_wandb.py script with the same config file
        # singularity exec \
        #     --overlay ${OVERLAY_EXT3}:ro \
        #     --overlay /scratch/work/public/singularity/vulkan-1.4.309-cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sqf:ro \
        #     /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
        #     /bin/bash -c "source ./set_path.sh; \
        #                 export PATH='/ext3/uv:$PATH'; \
        #                 source ./.venv/bin/activate; \
        #                 uv run scripts/eval/gather_data_to_wandb.py \
        #                 --config_path config/experiment/simpler/${CONFIG_NAME}"
    done
done
