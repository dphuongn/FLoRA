#!/bin/bash
# IID few-shot scenario
# FLoRA

cd ../../system/

nohup python -u main.py \
    -data caltech101 \
    -m lora \
    -algo flora \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -sd 0 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --lora_projection_text True \
    > result-caltech101-iid-fs-flora-npz.out 2>&1 & 