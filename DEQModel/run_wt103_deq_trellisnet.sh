#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (DEQ-TrellisNet)...'
    python train_trellisnet.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --n_layer 40 \
        --d_embed 500 \
        --nhid 1500 \
        --nout 500 \
        --clip 0.07 \
        --dropout 0.1 \
        --dropouti 0.1 \
        --dropouth 0.1 \
        --optim Adam \
        --lr 1e-3 \
        --pretrain_steps 30000 \
        --seq_len 100 \
        --subseq_len 50 \
        --f_thres 45 \
        --b_thres 45 \
        --batch_size 49 \
        --gpu0_bsz 7 \
        --multi_gpu \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Not supported yet'
else
    echo 'unknown argment 1'
fi
