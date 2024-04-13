# Practical non-IID scenario
# FLoRA

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
    > result-oxfordpets-dir-flora-npz.out 2>&1 & 