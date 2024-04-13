# IID few-shot scenario
# FLoRA

nohup python -u main.py \
    -data caltech101 \
    -m lora \
    -algo fedlora \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -fs True \
    -sd 0 \
    > result-caltech101-iid-fs-flora-npz.out 2>&1 & 