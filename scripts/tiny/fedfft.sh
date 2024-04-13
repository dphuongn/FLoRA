#!/bin/bash
# Pathological non-IID few-shot scenario
# FedFFT

cd ../../system/

nohup python -u main.py \
    -data pets \
    -m fft \
    -algo fedfft \
    -gr 50 \
    -did 0 \
    -nc 10 \
    -lbs 128 \
    -fs True \
    -sd 0 \
    > result-tiny-pat-fs-fedfft-npz.out 2>&1 & 