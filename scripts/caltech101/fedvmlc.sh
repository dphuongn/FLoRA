# IID few-shot scenario
# FedVM-LC

cd ../../system/

nohup python -u main.py \
    -data caltech101 \
    -m vmlc \
    -algo fedvmlc \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -fs True \
    -sd 0 \
    > result-caltech101-iid-fs-fedvmlc-npz.out 2>&1 & 