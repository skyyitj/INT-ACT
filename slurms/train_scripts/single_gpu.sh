#!/bin/bash
source ./slurms/train_scripts/set_path.sh
export PYTHONPATH=$PYTHONPATH:/home/lishuang/intact
# Test script with single GPU to isolate the issue

# set all the paths to environment variables
source ./slurms/train_scripts/set_path.sh

# Force single GPU
export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU=1
echo "NUM_GPU=$NUM_GPU"

# Run without torchrun for single GPU
python src/agent/run.py --config_path config/train/pi0_baseline_bridge.yaml
