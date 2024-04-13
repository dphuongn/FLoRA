# IID scenario
# FedAA

cd ../../system/

nohup python -u main.py \
    -data cifar10 \
    -m aa \
    -algo fedaa \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -fs True \
    -sd 0 \
    > result-cifar10-iid-fedaa-npz.out 2>&1 & 