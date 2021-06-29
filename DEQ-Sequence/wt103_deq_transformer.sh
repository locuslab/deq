#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training (DEQ-Transformer)...'
    python train_transformer.py \
        --cuda \
        --data ./data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --div_val 4 \
        --n_layer 2 \
        --eval_n_layer 24 \
        --d_embed 700 \
        --d_model 700 \
        --n_head 10 \
        --d_head 70 \
        --d_inner 48000 \
        --dropout 0.05 \
        --dropatt 0.0 \
        --optim Adam \
        --lr 0.00025 \
        --warmup_step 16000 \
        --pretrain_steps 32000 \
        --eval-interval 5000 \
        --max_step 300000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --wnorm \
        --f_solver anderson \
        --b_solver broyden \
        --stop_mode rel \
        --f_thres 30 \
        --b_thres 35 \
        --jac_loss_weight 0.0 \
        --jac_loss_freq 0.0 \
        --jac_incremental 0 \
        --batch_size 56 \
        --gpu0_bsz 14 \
        --multi_gpu \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Not supported yet'
else
    echo 'unknown argment 1'
fi