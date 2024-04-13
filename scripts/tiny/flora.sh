# Pathological non-IID few-shot scenario
# FLoRA

nohup python -u main.py \
    -data tiny \
    -m lora \
    -algo fedlora \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -fs True \
    -sd 0 \
    > result-tiny-pat-fs-flora-npz.out 2>&1 & 