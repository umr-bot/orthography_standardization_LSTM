lang=$1
hdd=$2
model_cnt=$3
bash evaluate_wrapper_2.sh checkpoints/$lang/softmax_regression/fold1_hdd_$hdd/model_$model_cnt
bash evaluate_wrapper_2.sh checkpoints/$lang/lstm_1/fold1_hdd_$hdd/model_$model_cnt
bash evaluate_wrapper_2.sh checkpoints/$lang/lstm_2/fold1_hdd_$hdd/model_$model_cnt
bash evaluate_wrapper_2.sh checkpoints/$lang/lstm_3/fold1_hdd_$hdd/model_$model_cnt
#bash evaluate_wrapper_2.sh checkpoints/$lang/lstm_1_dropout_0.1/fold1_hdd_$hdd/model_$model_cnt
#bash evaluate_wrapper_2.sh checkpoints/$lang/lstm_1_dropout_0.2/fold1_hdd_$hdd/model_$model_cnt
#bash evaluate_wrapper_2.sh checkpoints/$lang/lstm_1_dropout_0.3/fold1_hdd_$hdd/model_$model_cnt
#bash evaluate_wrapper_2.sh checkpoints/$lang/lstm_1_dropout_0.4/fold1_hdd_$hdd/model_$model_cnt
bash evaluate_wrapper_2.sh checkpoints/$lang/bi_lstm_1/fold1_hdd_$hdd/model_$model_cnt
bash evaluate_wrapper_2.sh checkpoints/$lang/bi_lstm_2/fold1_hdd_$hdd/model_$model_cnt
bash evaluate_wrapper_2.sh checkpoints/$lang/bi_lstm_3/fold1_hdd_$hdd/model_$model_cnt
