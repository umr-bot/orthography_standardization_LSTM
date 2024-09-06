To train use:

    python train.py --foldset_num <foldset_num> --model_num <model_num> --hidden_size <hidden_dim_size> --dropout <dropout_percentage> --model <model_name> --num_epochs <number_epochs> --data_dir <path_to_train_file> --checkpoints <path_to_save_model_checkpoints>

    foldset_num: Specific foldset to use train data from

    model_num: Epoch (of previously trained model) to start training from

    hidden_size: Size of hidden dimension layers of lstm

    dropout: Dropout percentage to use on all lstm layers
    
    model: Model name, eg lstm_1 where '1' represents that number of layers in network

    num_epochs: Number of epochs to train upto

    data_dir: Sath to a directory containining foldset folders with names foldset{1,2,3,4,5} as seen in skeleton folder 'char_level_lstm/data'. Each foldset folder should contain a train, val and test file.

    checkpoints: Directory in which to save checkpoint models.

To evaluate use:
    
    python evaluate.py --foldset_num <foldset_num> --data_dir <path_to_train_file> --hidden_size hidden_dim_size> --model <model_name> --model_num <epoch_of_model_to_evaluate> --checkpoints_dir <path_to_save_model_checkpoints> --train_batch_size <train_batch_size> --val_batch_size <val_batch_size> --val_or_test <validation_or_test>

    foldset_num: Specific foldset to use train data from

    data_dir: Sath to a directory containining foldset folders with names foldset{1,2,3,4,5} as seen in skeleton folder 'char_level_lstm/data'. Each foldset folder should contain a train, val and test file.

    hidden_size: Size of hidden dimension layers of lstm

    model: Model name, eg lstm_1 where '1' represents that number of layers in network

    model_num: Epoch (of previously trained model) to start training from

    checkpoints: Directory in which to save checkpoint models.

    train_batch_size: Size of mini batches used in training
    
    val_batch_size: Size of mini batches used in validation

To compute metrics for a single fold use:

    python comp.py --errfile $root_dir/err_file --clnfile $root_dir/cln_file --tarfile $root_dir/tar_file >> $root_dir/metrics.txt

    $root_dir: a folder named {val,test}_model_xx generated in the checkpoints directory by the evaluate.py script
----------------------------------------------------------------------------------------------

Example of train, evaluate and metric computation:

    python train.py --foldset_num 1 --model_num 0 --hidden_size 512 --dropout 0 --model "lstm_1" --num_epochs 10 --data_dir data/unigrams --checkpoints checkpoints/data/lstm_1

    python evaluate.py --foldset_num 1 --data_dir data/unigrams --hidden_size 512 --model "lstm_1" --model_num 10 --checkpoints_dir checkpoints/data/lstm_1 --train_batch_size 10 --val_batch_size 10 --val_or_test "val"
 
    python comp.py --errfile checkpoints/data/lstm_1/foldset_1_hdd_512/val_model_10/err_file --clnfile checkpoints/data/lstm_1/foldset_1_hdd_512/val_model_10/cln_file --tarfile checkpoints/data/lstm_1/foldset_1_hdd_512/val_model_10/tar_file >> checkpoints/data/lstm_1/foldset_1_hdd_512/val_model_10/metrics.txt
