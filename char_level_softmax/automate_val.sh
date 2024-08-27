#!/bin/bash
for i in 1 2 3 4 5; do
    python evaluate.py --hidden_size -1 --foldset_num ${i} --data_dir bam_v3/unigrams --model_num 10 --checkpoints_dir checkpoints/bam/foldset_${i} --val_or_test "test"
done

for i in 1 2 3 4 5; do
    python evaluate.py --hidden_size -1 --foldset_num ${i} --data_dir eng_v2/unigrams --model_num 10 --checkpoints_dir checkpoints/eng/foldset_${i} --val_or_test "test"
done

for i in 1 2 3 4 5; do
    python evaluate.py --hidden_size -1 --foldset_num ${i} --data_dir eng_v2/unigrams_reduced_v1 --model_num 10 --checkpoints_dir checkpoints/eng_subset_v1/foldset_${i} --val_or_test "test"
done
for i in 1 2 3 4 5; do
    python evaluate.py --hidden_size -1 --foldset_num ${i} --data_dir eng_v2/unigrams_reduced_v2 --model_num 10 --checkpoints_dir checkpoints/eng_subset_v2/foldset_${i} --val_or_test "test"
done
