from keras.models import load_model
from model import precision,recall,f1_score
from utils import CharacterTable
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Logistic regression training script")
parser.add_argument("--num_epochs", default=20, help="hidden layer1 size")
parser.add_argument("--train_batch_size", default=128, help="train batch size")
parser.add_argument("--val_batch_size", default=256, help="val batch size")
parser.add_argument("--foldset_num", default=1, help="foldset number to use")

args = parser.parse_args()
args.num_epochs, args.train_batch_size, args.val_batch_size = int(args.num_epochs), int(args.train_batch_size), int(args.val_batch_size)

#foldset_num = 1
with open("eng_v2/unigrams/foldset"+str(args.foldset_num)+"/train") as f:
    train_tups = [line.strip('\n').split(',') for line in f]
train_dec_tokens, train_tokens = zip(*train_tups)
with open("eng_v2/unigrams/foldset"+str(args.foldset_num)+"/val") as f:
    val_tups = [line.strip('\n').split(',') for line in f]
val_dec_tokens, val_tokens = zip(*val_tups)

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
total_chars = input_chars.union(target_chars)
nb_total_chars = len(total_chars)
total_ctable = input_ctable = target_ctable = CharacterTable(total_chars)
maxlen = max([len(token) for token in train_tokens]) + 2

custom_objects = { 'recall': recall, "precision": precision, "f1_score": f1_score}
model = load_model("checkpoints/eng_checkpoints_lstm_1/seq2seq_epoch_5.h5",custom_objects=custom_objects)

word_batch=np.zeros((///))
word=total_ctable.encode("hel",nb_rows=maxlen)
print(total_ctable.decode(word,calc_argmax=True))
word_batch=word[np.newaxis,...]

word_y=model.predict(word_batch)
print(total_ctable.decode(word_y[0],calc_argmax=True))
