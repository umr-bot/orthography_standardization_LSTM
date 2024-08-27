# coding: utf-8
import os
# NOTE: log levels for tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model, load_model
from tqdm import tqdm
from utils import CharacterTable, batch, datagen_simple, transform_3
import argparse

import tensorflow as tf
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.333
session = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser(description="Logistic regression training script")
parser.add_argument("--hidden_size", default=128, help="hidden layer1 size")
parser.add_argument("--train_batch_size", default=10, help="train batch size")
parser.add_argument("--val_batch_size", default=100, help="val batch size")
parser.add_argument("--num_epochs", default=20, help="number of epochs")
parser.add_argument("--foldset_num", default=1, help="foldset number to use")
parser.add_argument("--checkpoints_dir", default="eng_checkpoints", help="directory in which to save checkpoints")
parser.add_argument("--data_dir",help="path to unigrams")
#parser.add_argument("--lang",help="language being trained on")
parser.add_argument("--model",help="model name")
parser.add_argument("--model_num",help="which model version is being evaluated on")
parser.add_argument("--batch_size", default=100, help="size of batches to use in inference")
parser.add_argument("--val_or_test", help="whether to load val or test datasets")

args = parser.parse_args()
args.hidden_size, args.num_epochs, args.train_batch_size,args.val_batch_size=int(args.hidden_size),int(args.num_epochs),int(args.train_batch_size), int(args.val_batch_size)
#assert(len(int(args.val_batch_size)) < len(int(args.)))

with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)

with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/"+args.val_or_test) as f:
    val_tups = [line.strip('\n').split(',') for line in f]
val_dec_tokens, val_tokens = zip(*val_tups)

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
total_chars = input_chars.union(target_chars)
nb_total_chars = len(total_chars)
total_ctable = input_ctable = target_ctable = CharacterTable(total_chars)
maxlen = max([len(token) for token in train_tokens]) + 2
#########################################################################################

# Load model
root_dir=args.checkpoints_dir+'/foldset_' + str(args.foldset_num)
with session.as_default():
    model = load_model( root_dir + "/"+args.model+"_epoch_"+args.model_num+".h5")
    #########################################################################################

    val_x, val_y_pred,val_y_true=[],[],[]
    val_x_padded=transform_3(val_tokens, maxlen=maxlen)
    val_y_padded=transform_3(val_dec_tokens, maxlen=maxlen)

    # when batch_size=1 loads 1 sample at a time
    batch_size=int(args.batch_size)
    val_X_iter = batch(val_x_padded,maxlen=maxlen,ctable=total_ctable,batch_size=batch_size,reverse=False)
    val_y_iter = batch(val_y_padded,maxlen=maxlen,ctable=total_ctable,batch_size=batch_size,reverse=False)
    val_loader = datagen_simple(val_X_iter, val_y_iter)

    for unused_cnt in tqdm(range(0, len(val_tokens), batch_size), desc=f"Computing predictions"):
        val_batch = next(val_loader)
        # Get one sample from batch of validation data
        x,y=val_batch[0],val_batch[1]        
        # model.predict computes predictions for batches of samples, batch_size is min at 1 sample
        y_pred = model.predict(x, verbose=0)
        # uncomment line below to loop over samples if using batch_size > 1 and change y_pred[0] to y_pred[word_ind]
        for word_ind in range(batch_size):
            val_x.append(total_ctable.decode(x[word_ind], calc_argmax=True)[1])
            val_y_pred.append(total_ctable.decode(y_pred[word_ind], calc_argmax=True)[1])
            val_y_true.append(total_ctable.decode(y[word_ind], calc_argmax=True)[1])

    save_dir = root_dir+"/"+args.val_or_test+"_model_"+args.model_num
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir+"/err_file",'w') as f:
        for word in val_x: f.write(word+'\n')
    with open(save_dir+"/cln_file",'w')as f:
        for word in val_y_pred: f.write(word+'\n')
    with open(save_dir+"/tar_file",'w') as f:
        for word in val_y_true: f.write(word+'\n')

