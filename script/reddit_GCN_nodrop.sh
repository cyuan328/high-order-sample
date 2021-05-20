#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 99 \
    --dataset reddit \
    --type mutigcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 32 \
    --hidden 128 \
    --epoch 2000 \
    --lr 0.01 \
    --weight_decay 0 \
    --early_stopping 1000 \
    --sampling_percent 1 \
    --dropout 0 \
    --normalization BingGeNormAdj \
    --withloop \
    --withbn
