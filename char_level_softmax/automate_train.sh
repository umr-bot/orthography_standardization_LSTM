#!bin/bash
for i in 1 2 3 4 5; do
    python train.py --foldset_num ${i} --model_num 0 --hidden_size -1 --dropout -1 --model "softmax_regression" --num_epochs 10 --data_dir bam_v3/unigrams --checkpoints checkpoints/bam/foldset_${i} --use_bigrams 'false' --val_or_test "val"
done

for i in 1 2 3 4 5; do
    python train.py --foldset_num ${i} --model_num 0 --hidden_size -1 --dropout -1 --model "softmax_regression" --num_epochs 10 --data_dir eng_v2/unigrams --checkpoints checkpoints/eng --use_bigrams 'false' --val_or_test "val"
done
for i in 1 2 3 4 5; do
    python train.py --foldset_num ${i} --model_num 0 --hidden_size -1 --dropout -1 --model "softmax_regression" --num_epochs 10 --data_dir eng_v2/unigrams_reduced_v1 --checkpoints checkpoints/eng_subset_v1 --use_bigrams 'false' --val_or_test "val"
done
for i in 1 2 3 4 5; do
    python train.py --foldset_num ${i} --model_num 0 --hidden_size -1 --dropout -1 --model "softmax_regression" --num_epochs 10 --data_dir eng_v2/unigrams_reduced_v2 --checkpoints checkpoints/eng_subset_v2 --use_bigrams 'false' --val_or_test "val"
done

