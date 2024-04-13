#!/bin/bash
# Practical non-IID scenario
# FLoRA

cd ../../system/

nohup python -u main.py \
    -data pets \
    -m lora \
    -algo fedlora \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -fs True \
    -sd 0 \
    --lora_rank 2 \
    --lora_alpha 32 \
    --lora_projection_text True \
    > result-oxfordpets-dir-flora-npz.out 2>&1 & 