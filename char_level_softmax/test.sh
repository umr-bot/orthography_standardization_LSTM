#!/bin/bash
b_l1=0.01
b_l2=0.01
model_start_num=0
m_num=10
for k_l1 in 0.0001 0.001 0.01; do
    for k_l2 in 0.0001 0.001 0.01; do
        for i in 2 3 4 5; do
            python train_reg.py --foldset_num ${i} --model_num ${model_start_num} --hidden_size -1 --dropout -1 --model "softmax_regression_reg" --num_epochs ${m_num} --data_dir eng_v2/unigrams --checkpoints checkpoints/eng_reg/b_l1_${b_l1}_b_l2_${b_l2}_k_l1_${k_l1}_k_l2_${k_l2} --bias_l1 ${b_l1} --bias_l2 ${b_l2} --kernel_l1 ${k_l1} --kernel_l2 ${k_l2}
            python evaluate.py --foldset_num ${i} --model_num ${m_num} --data_dir eng_v2/unigrams --checkpoints checkpoints/eng_reg/b_l1_${b_l1}_b_l2_${b_l2}_k_l1_${k_l1}_k_l2_${k_l2} --model "softmax_regression_reg" --val_or_test "val"
            python comp.py --errfile checkpoints/eng_reg/b_l1_${b_l1}_b_l2_${b_l2}_k_l1_${k_l1}_k_l2_${k_l2}/foldset_${i}/val_model_${m_num}/err_file --clnfile checkpoints/eng_reg/b_l1_${b_l1}_b_l2_${b_l2}_k_l1_${k_l1}_k_l2_${k_l2}/foldset_${i}/val_model_${m_num}/cln_file --tarfile checkpoints/eng_reg/b_l1_${b_l1}_b_l2_${b_l2}_k_l1_${k_l1}_k_l2_${k_l2}/foldset_${i}/val_model_${m_num}/tar_file > checkpoints/eng_reg/b_l1_${b_l1}_b_l2_${b_l2}_k_l1_${k_l1}_k_l2_${k_l2}/foldset_${i}/val_model_${m_num}/metrics.txt
        done
    done
done
