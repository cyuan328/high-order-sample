#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 88 \
    --dataset reddit \
    --type inceptiongcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 16 \
    --hidden 128 \
    --epoch 2000 \
    --lr 0.002 \
    --weight_decay 0.005 \
    --early_stopping 1000 \
    --sampling_percent 1 \
    --dropout 0.5 \
    --normalization BingGeNormAdj \
    --withloop \
    
