#!/bin/bash

MODELS=("GCN")
MODES=("samplingInference")
DATASETS=("Cora" "Flickr" "Reddit2" "Products")
# MODELS=("GCN" "SAGE")
# MODES=("fullInference" "samplingInference")
RUNS=2
DEVICE="CPU"

# 三层循环遍历运行 Python 脚本
for MODE in "${MODES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for DATA in "${DATASETS[@]}"; do
            echo "Running with model: $MODEL, data: $DATA, mode: $MODE"
            python3 CPU_Inference.py --model "$MODEL" --data "$DATA" --mode "$MODE" --runs $RUNS --device "$DEVICE" 
        done
    done
done

echo "Bash Done!"
