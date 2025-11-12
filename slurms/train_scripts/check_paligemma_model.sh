#!/bin/bash
# Script to check if PaliGemma model exists locally

TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/media/jushen/linda-zhao/data1/open-pi-zero/transformer_cache}"
MODEL_PATH="${1:-${TRANSFORMERS_CACHE}/paligemma-3b-pt-224}"

echo "Checking for PaliGemma model at: $MODEL_PATH"

# Check if path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Path does not exist: $MODEL_PATH"
    exit 1
fi

# Check for model files
HAS_SAFETENSORS=false
HAS_PYTORCH_BIN=false
HAS_CONFIG=false

if ls "$MODEL_PATH"/*.safetensors 1> /dev/null 2>&1; then
    HAS_SAFETENSORS=true
    echo "✅ Found .safetensors files"
fi

if [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
    HAS_PYTORCH_BIN=true
    echo "✅ Found pytorch_model.bin"
fi

if [ -f "$MODEL_PATH/config.json" ]; then
    HAS_CONFIG=true
    echo "✅ Found config.json"
fi

# Summary
if [ "$HAS_SAFETENSORS" = true ] || [ "$HAS_PYTORCH_BIN" = true ]; then
    if [ "$HAS_CONFIG" = true ]; then
        echo ""
        echo "✅ Model files found! Model is ready to use."
        exit 0
    else
        echo ""
        echo "⚠️  Model weight files found but config.json is missing."
        exit 1
    fi
else
    echo ""
    echo "❌ No model weight files found (.safetensors or pytorch_model.bin)"
    echo "   The model needs to be downloaded or the path is incorrect."
    exit 1
fi

