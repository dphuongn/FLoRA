# Practical non-IID scenario
# FedLC

nohup python -u main.py \
    -data pets \
    -m lc \
    -algo fedlc \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -fs True \
    -sd 0 \
    > result-oxfordpets-dir-fedlc-npz.out 2>&1 & 