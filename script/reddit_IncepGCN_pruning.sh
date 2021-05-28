#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 88 \
    --dataset reddit \
    --type inceptiongcn \
    --nhiddenlayer 1 \
    --nbaseblocklayer 2 \
    --hidden 128 \
    --epoch 4000 \
    --lr 0.002 \
    --weight_decay 0.005 \
    --early_stopping 2000 \
    --sampling_percent 0.2 \
    --dropout 0.5 \
    --normalization BingGeNormAdj \
    --withloop \
    --sampling_method pruning 
