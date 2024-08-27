#!/bin/bash
for i in 1 2 3 4 5; do
    python comp.py --errfile checkpoints/bam/foldset_${i}/val_model_10/err_file --clnfile checkpoints/bam/foldset_${i}/val_model_10/cln_file --tarfile checkpoints/bam/foldset_${i}/val_model_10/tar_file > checkpoints/bam/foldset_${i}/val_model_10/metrics.txt
done
for i in 1 2 3 4 5; do
    python comp.py --errfile checkpoints/bam/foldset_${i}/test_model_10/err_file --clnfile checkpoints/bam/foldset_${i}/test_model_10/cln_file --tarfile checkpoints/bam/foldset_${i}/test_model_10/tar_file > checkpoints/bam/foldset_${i}/test_model_10/metrics.txt
done

for i in 1 2 3 4 5; do
    python comp.py --errfile checkpoints/eng/foldset_${i}/val_model_10/err_file --clnfile checkpoints/eng/foldset_${i}/val_model_10/cln_file --tarfile checkpoints/eng/foldset_${i}/val_model_10/tar_file > checkpoints/eng/foldset_${i}/val_model_10/metrics.txt
done
for i in 1 2 3 4 5; do
    python comp.py --errfile checkpoints/eng/foldset_${i}/test_model_10/err_file --clnfile checkpoints/eng/foldset_${i}/test_model_10/cln_file --tarfile checkpoints/eng/foldset_${i}/test_model_10/tar_file > checkpoints/eng/foldset_${i}/test_model_10/metrics.txt
done

for i in 1 2 3 4 5; do
    python comp.py --errfile checkpoints/eng_subset_v1/foldset_${i}/val_model_10/err_file --clnfile checkpoints/eng_subset_v1/foldset_${i}/val_model_10/cln_file --tarfile checkpoints/eng_subset_v1/foldset_${i}/val_model_10/tar_file > checkpoints/eng_subset_v1/foldset_${i}/val_model_10/metrics.txt
done
for i in 1 2 3 4 5; do
    python comp.py --errfile checkpoints/eng_subset_v1/foldset_${i}/test_model_10/err_file --clnfile checkpoints/eng_subset_v1/foldset_${i}/test_model_10/cln_file --tarfile checkpoints/eng_subset_v1/foldset_${i}/test_model_10/tar_file > checkpoints/eng_subset_v1/foldset_${i}/test_model_10/metrics.txt
done

for i in 1 2 3 4 5; do
    python comp.py --errfile checkpoints/eng_subset_v2/foldset_${i}/val_model_10/err_file --clnfile checkpoints/eng_subset_v2/foldset_${i}/val_model_10/cln_file --tarfile checkpoints/eng_subset_v2/foldset_${i}/val_model_10/tar_file > checkpoints/eng_subset_v2/foldset_${i}/val_model_10/metrics.txt
done
for i in 1 2 3 4 5; do
    python comp.py --errfile checkpoints/eng_subset_v2/foldset_${i}/test_model_10/err_file --clnfile checkpoints/eng_subset_v2/foldset_${i}/test_model_10/cln_file --tarfile checkpoints/eng_subset_v2/foldset_${i}/test_model_10/tar_file > checkpoints/eng_subset_v2/foldset_${i}/test_model_10/metrics.txt
done

