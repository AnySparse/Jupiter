#!/bin/bash

BIN_PATH=./build/
GRAPH_PATH=/data/linzhiheng/

PATTERNS=("P2" "P5" "P9" "P13" "P14")

DATASETS=("youtube" "friendster" "datagen-sf10k-fb")

MPIRUN_ARGS="mpirun -n 4 --bind-to numa"

for pattern in "${PATTERNS[@]}"; do
    echo "Pattern $pattern Jupiter runtime on YO,FR,DG:"
    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" = "youtube" ]; then
            total_round=1
        else
            total_round=10
        fi
        CUDA_VISIBLE_DEVICES=4,5,6,7 NVSHMEM_SYMMETRIC_SIZE=20000000000 $MPIRUN_ARGS $BIN_PATH/jupiter_gpm $GRAPH_PATH$dataset/graph $pattern $total_round | grep "runtime:" | grep -oP '\d*\.\d+'
    done
done
