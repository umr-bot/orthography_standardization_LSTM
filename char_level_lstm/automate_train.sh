#!/bin/bash

for hdd in 256 512 1024; do
    for num_layer in 1 2 3; do
        for i in 1 2 3 4 5; do
            python train.py --foldset_num ${i} --model_num 0 --hidden_size ${hdd} --dropout 0 --model "lstm_${num_layer}" --num_epochs 10 --data_dir bam_v3/unigrams --checkpoints checkpoints/bam/lstm_${num_layer}
        done
    done
done

for hdd in 256 512 1024; do
    for num_layer in 1 2 3; do
        for i in 1 2 3 4 5; do
            python train.py --foldset_num ${i} --model_num 0 --hidden_size ${hdd} --dropout 0 --model "lstm_${num_layer}" --num_epochs 10 --data_dir eng_v2/unigrams --checkpoints checkpoints/eng/lstm_${num_layer}
        done
    done
done

for hdd in 256 512 1024; do
    for num_layer in 1 2 3; do
        for i in 1 2 3 4 5; do
            python train.py --foldset_num ${i} --model_num 0 --hidden_size ${hdd} --dropout 0 --model "lstm_${num_layer}" --num_epochs 10 --data_dir eng_v2/unigrams_reduced_v1 --checkpoints checkpoints/eng_subset_v1/lstm_${num_layer}
        done
    done
done
for hdd in 256 512 1024; do
    for num_layer in 1 2 3; do
        for i in 1 2 3 4 5; do
            python train.py --foldset_num ${i} --model_num 0 --hidden_size ${hdd} --dropout 0 --model "lstm_${num_layer}" --num_epochs 10 --data_dir eng_v2/unigrams_reduced_v2 --checkpoints checkpoints/eng_subset_v2/lstm_${num_layer}
        done
    done
done

