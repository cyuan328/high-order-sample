#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 77 \
    --dataset reddit \
    --type densegcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 16 \
    --hidden 128 \
    --epoch 2000 \
    --lr 0.004 \
    --weight_decay 5e-05 \
    --early_stopping 1000 \
    --sampling_percent 0.6 \
    --dropout 0.3 \
    --normalization AugNormAdj \
    --withloop \
    
