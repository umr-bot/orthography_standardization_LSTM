#!/bin/bash
num_epochs=10
train_batch_size=128
val_batch_size=256
fold_num=1
model="lstm_basic_3" #"lstm_basic" 
lang="eng" # make sure to change lang_version appropriately
lang_version="2"
# English #############################################################################################
# reset metric files to empty
###cat /dev/null >| checkpoints/eng/$model/fold${fold_num}_hdd_512/model_${num_epochs}/metrics.txt
###cat /dev/null >|  checkpoints/eng/$model/fold${fold_num}_hdd_1028/model_${num_epochs}/metrics.txt
###for hdd in "512" "1028"; do
###    python evaluate.py --foldset_num $fold_num --hidden_size $hdd --num_epochs $num_epochs --data_dir eng_v2/unigrams --model_num $num_epochs --checkpoints_dir checkpoints/eng/$model --train_batch_size $train_batch_size --val_batch_size $val_batch_size
###    echo eng_fold_${fold_num}_hdd_${hdd}_num_epochs_${num_epochs}_train_batch_size_${train_batch_size}_val_batch_size_${val_batch_size} >> checkpoints/eng/$model/fold${fold_num}_hdd_${hdd}/model_$num_epochs/metrics.txt
###    python comp.py --errfile checkpoints/eng/$model/fold${fold_num}_hdd_$hdd/model_$num_epochs/err_file --clnfile checkpoints/eng/$model/fold${fold_num}_hdd_${hdd}/model_$num_epochs/cln_file --tarfile checkpoints/eng/$model/fold${fold_num}_hdd_${hdd}/model_$num_epochs/tar_file >> checkpoints/eng/$model/fold${fold_num}_hdd_${hdd}/model_$num_epochs/metrics.txt
###done
#######################################################################################################
# Bambara
# reset metric files to empty
mkdir -p checkpoints/$lang/$model/fold${fold_num}_hdd_512/model_${num_epochs}/
mkdir -p checkpoints/$lang/$model/fold${fold_num}_hdd_1028/model_${num_epochs}/
cat /dev/null | tee checkpoints/$lang/$model/fold${fold_num}_hdd_512/model_${num_epochs}/{metrics.txt,err_file,cln_file,tar_file}
cat /dev/null | tee  checkpoints/$lang/$model/fold${fold_num}_hdd_1028/model_${num_epochs}/{metrics.txt,err_file,cln_file,tar_file}
for hdd in "512" "1028"; do
    python evaluate.py --foldset_num $fold_num --hidden_size $hdd --num_epochs $num_epochs --data_dir ${lang}_v${lang_version}/unigrams --model_num $num_epochs --checkpoints_dir checkpoints/$lang/$model --train_batch_size $train_batch_size --val_batch_size $val_batch_size
    echo $lang_fold_${fold_num}_hdd_${hdd}_num_epochs_${num_epochs}_train_batch_size_${train_batch_size}_val_batch_size_${val_batch_size} >> checkpoints/$lang/$model/fold${fold_num}_hdd_${hdd}/model_$num_epochs/metrics.txt
    python comp.py --errfile checkpoints/$lang/$model/fold${fold_num}_hdd_$hdd/model_$num_epochs/err_file --clnfile checkpoints/$lang/$model/fold${fold_num}_hdd_${hdd}/model_$num_epochs/cln_file --tarfile checkpoints/$lang/$model/fold${fold_num}_hdd_${hdd}/model_$num_epochs/tar_file >> checkpoints/$lang/$model/fold${fold_num}_hdd_${hdd}/model_$num_epochs/metrics.txt
done
