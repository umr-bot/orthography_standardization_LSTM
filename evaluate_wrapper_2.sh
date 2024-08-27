#!/bin/bash
#root_dir="checkpoints/eng/lstm_basic_1/fold1_hdd_1028/model_10"
root_dir=$1
metric_header=$(sed 's/\//_/g' <<< $root_dir)
mkdir -p $1 # will create the directory if it doesn't exist
echo $metric_header > $1/metrics.txt
python comp.py --errfile $root_dir/err_file --clnfile $root_dir/cln_file --tarfile $root_dir/tar_file >> $root_dir/metrics.txt

