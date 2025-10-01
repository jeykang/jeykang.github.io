#!/usr/bin/env bash
#--dataset_root /jisu/dataset/iHarmony4resized256/ \

python train.py \
--dataset_root /mnt/CA3EAE403EAE2603/tempdown/Image_Harmonization_Dataset/processed \
--name no_RAIN_only_IN \
--checkpoints_dir ./checkpoints/no_RAIN_only_IN/ \
--model rainnet \
--netG rainnet \
--dataset_mode iharmony4 \
--is_train 1 \
--gan_mode wgangp \
--normD instance \
--normG RAIN \
--preprocess None \
--niter 100 \
--niter_decay 100 \
--input_nc 3 \
--batch_size 11 \
--num_threads 6 \
--lambda_L1 100 \
--print_freq 400 \
--gpu_ids 0 \
#--continue_train \
#--load_iter 87 \
#--epoch 88 \
