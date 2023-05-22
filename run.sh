#!/usr/bin bash
for s in 42
do
      python DeepAligned.py \
        --dataset banking \
        --known_cls_ratio 0.75 \
        --cluster_num_factor 1 \
        --seed $s \
        --gpu_id 2\
        --beta 0.6\
        --lr 5e-5\
        --lr_pre 5e-5\
        --train_batch_size 512\
        --name banking_k1_bz512_lr5e-5_lrp5e-5_seed${s}_aug2_augp\
        --freeze_bert_parameters_EM\
        --freeze_bert_parameters_pretrain\
        --pretrain\
        --k 1\
        --augment_data
done


