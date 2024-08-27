#!/bin/bash
# English
model=$1
dropout=$2
###python train.py --hidden_size 512 --model 'lr' --num_epochs 10 --data_dir eng_v2/unigrams --lang "eng" --checkpoints checkpoints/eng/softmax_regression --train_batch_size 128 --val_batch_size 256 && python train.py --hidden_size 1028 --model 'lr' --num_epochs 10 --data_dir eng_v2/unigrams --lang "eng" --checkpoints checkpoints/eng/softmax_regression --train_batch_size 128 --val_batch_size 256
###echo "eng softmax regression done training"
###
######python train.py --hidden_size 128 --model 'lstm' --num_epochs 10 --data_dir eng_v2/unigrams --lang "eng" --checkpoints checkpoints/eng/lstm_basic --train_batch_size 128 --val_batch_size 256 && 
###python train.py --hidden_size 512 --model 'lstm' --num_epochs 10 --data_dir eng_v2/unigrams --lang "eng" --checkpoints checkpoints/eng/lstm_basic --train_batch_size 128 --val_batch_size 256 && python train.py --hidden_size 1028 --model 'lstm' --num_epochs 10 --data_dir eng_v2/unigrams --lang "eng" --checkpoints checkpoints/eng/lstm_basic --train_batch_size 128 --val_batch_size 256
###echo "eng lstm done training"
#######################################################################################################
# Bambara
#python train.py --hidden_size 128 --model 'lr' --num_epochs 10 --data_dir bam_v3/unigrams --lang "bam" --checkpoints checkpoints/bam/softmax_regression --train_batch_size 128 --val_batch_size 256 && 
#python train.py --hidden_size 512 --model 'lr' --num_epochs 10 --data_dir bam_v3/unigrams --lang "bam" --checkpoints checkpoints/bam/softmax_regression --train_batch_size 128 --val_batch_size 256 && python train.py --hidden_size 1028 --model 'lr' --num_epochs 10 --data_dir bam_v3/unigrams --lang "bam" --checkpoints checkpoints/bam/softmax_regression --train_batch_size 128 --val_batch_size 256
#echo "bam softmax regression done training"

###python train.py --hidden_size 128 --model 'lstm' --num_epochs 10 --data_dir bam_v3/unigrams --lang "bam" --checkpoints checkpoints/bam/lstm_basic --train_batch_size 128 --val_batch_size 256 && 
#python train.py --hidden_size 512 --model 'lstm' --num_epochs 10 --data_dir bam_v3/unigrams --lang "bam" --checkpoints checkpoints/bam/lstm_basic --train_batch_size 128 --val_batch_size 256 && python train.py --hidden_size 1028 --model 'lstm' --num_epochs 10 --data_dir bam_v3/unigrams --lang "bam" --checkpoints checkpoints/bam/lstm_basic --train_batch_size 128 --val_batch_size 256
#echo "bam lstm done training"

#######################################################################################################
mkdir -p checkpoints/eng/$model
mkdir -p checkpoints/bam/$model
#python train.py --hidden_size 512 --model $model --num_epochs 10 --data_dir eng_v2/unigrams --lang "eng" --checkpoints checkpoints/eng/$model --train_batch_size 128 --val_batch_size 256 && 
python train.py --hidden_size 1028 --model $model --num_epochs 10 --data_dir eng_v2/unigrams --lang "eng" --checkpoints checkpoints/eng/$model --train_batch_size 128 --val_batch_size 256 --dropout $dropout
#python train.py --hidden_size 512 --model $model --num_epochs 10 --data_dir bam_v3/unigrams --lang "bam" --checkpoints checkpoints/bam/$model --train_batch_size 128 --val_batch_size 256 &&
python train.py --hidden_size 1028 --model $model --num_epochs 10 --data_dir bam_v3/unigrams --lang "bam" --checkpoints checkpoints/bam/$model --train_batch_size 128 --val_batch_size 256 --dropout $dropout

date
