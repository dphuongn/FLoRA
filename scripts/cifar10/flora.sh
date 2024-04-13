# IID scenario
# FLoRA

nohup python -u main.py \
    -data cifar10 \
    -m lora \
    -algo fedlora \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -fs True \
    -sd 0 \
    > result-cifar10-iid-flora-npz.out 2>&1 & 