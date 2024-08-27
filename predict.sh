#!/bin/bash
# $1 = checkpoints version
python evaluate.py --checkpoints_dir "eng_checkpoints_$1" --train_batch_size 256 --val_batch_size 256
python comp.py --errfile eng_checkpoints_$1/err_file --clnfile eng_checkpoints_$1/cln_file --tarfile eng_checkpoints_$1/tar_file

