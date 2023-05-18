#!/usr/bin bash

for s in 42 52
do
      python DeepAligned.py \
        --dataset clinc \
        --known_cls_ratio 0.75 \
        --cluster_num_factor 4 \
        --seed $s \
        --gpu_id 1\
        --beta 0.6\
        --lr 5e-5\
        --lr_ft 5e-5\
        --train_batch_size 512\
        --name clinc_k1_bz512_lr5e-5_lrp1e-5_lrf5e-5_seed${s}_aug2_beta0.6_cnf4\
        --freeze_bert_parameters_em\
        --freeze_bert_parameters_pretrain\
        --k 1\
        --augment_data_2\
        --pretrain\

done


