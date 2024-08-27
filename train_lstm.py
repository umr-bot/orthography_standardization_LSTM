# coding: utf-8
import os
from utils import CharacterTable, transform2
from utils_2 import train_generator_2
from models import lstm_model
import pickle
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
#from keras.optimizers import RMSprop

import argparse

parser = argparse.ArgumentParser(description="Logistic regression training script")
parser.add_argument("--hidden_size", default=128, help="hidden layer1 size")
parser.add_argument("--num_epochs", default=20, help="hidden layer1 size")
parser.add_argument("--train_batch_size", default=128, help="train batch size")
parser.add_argument("--val_batch_size", default=256, help="val batch size")
parser.add_argument("--foldset_num", default=1, help="foldset number to use")
parser.add_argument("--checkpoints_dir", default="eng_checkpoints", help="directory in which to save checkpoints")
parser.add_argument("--history_fn", default="history.txt", help="file name to save model history in")

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

#train_encoder, train_decoder, train_target = transform2( train_tokens, maxlen, shuffle=False, dec_tokens=train_dec_tokens)
#tg = train_generator_2(train_encoder,train_decoder,maxlen=maxlen,ctable=total_ctable,batch_size=batch_size)
#tg = train_generator_2(train_tokens,train_dec_tokens,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size)
#vg = train_generator_2(val_tokens,val_dec_tokens,maxlen=maxlen,ctable=total_ctable,batch_size=args.val_batch_size)
###batch_size=5
###for b in range(batch_size):
###    batch = next(tg)
###    x,y = batch[0],batch[1]
###    for i in range(len(y)):
###        print(total_ctable.decode(y[i], calc_argmax=True)[1])

train_steps = len(train_tokens) // args.num_epochs
val_steps = len(val_tokens) // args.num_epochs

model = lstm_model(hidden_size=256,nb_features=total_ctable.size)

start=0
for epoch in range(start,args.num_epochs):
    print(f"Epoch {str(epoch+1)}")
    st_ind,et_ind = int(args.train_batch_size*(epoch/args.num_epochs)), int(args.train_batch_size*((epoch+1)/args.num_epochs) )
    sv_ind,ev_ind = int(args.val_batch_size*(epoch/args.num_epochs)), int(args.val_batch_size*((epoch+1)/args.num_epochs))

    tg = train_generator_2(train_tokens[st_ind:et_ind],train_dec_tokens[st_ind:et_ind],maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size)
    vg = train_generator_2(val_tokens[sv_ind:ev_ind],val_dec_tokens[sv_ind:ev_ind],maxlen=maxlen,ctable=total_ctable,batch_size=args.val_batch_size)

    #train_loader=datagen(tg)
    #val_loader=datagen(vg)
    history = model.fit(tg, steps_per_epoch=train_steps, validation_data=vg, validation_steps=val_steps, epochs=1, verbose=1)

    # Save the model at end of each epoch.
    model_file = '_'.join(['seq2seq', 'epoch', str(epoch + 1)]) + '.h5'
    save_dir = args.checkpoints_dir #'eng_checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if (epoch+1) % 5 == 0:
        save_path = os.path.join(save_dir, model_file)
        print('Saving full model to {:s}'.format(save_path))
        model.save(save_path)
    fn = save_dir+ '/' + args.history_fn #"/history.txt"
    #for history in score:
    with open(fn,'a') as f:
        f.write(str(history.history['loss'][0]) + ',')
        f.write(str(history.history['val_loss'][0]) + ',')
        f.write(str(history.history['accuracy'][0]) + ',')
        f.write(str(history.history['val_accuracy'][0]))
        #f.write(str(history.history['precision'][0]) + ',')
        #f.write(str(history.history['recall'][0]) + ',')
        #f.write(str(history.history['f1_score'][0])+',')
        #f.write(str(history.history['val_precision'][0]) + ',')
        #f.write(str(history.history['val_recall'][0]) + ',')
        #f.write(str(history.history['val_f1_score'][0]))
       
        f.write('\n')


#model.save('lstm_next_character_model.h5')
#pickle.dump(history, open(args.eng_checkpoints+"history.p", "wb"))

#model = load_model('next_word_model.h5')
#history = pickle.load(open("history.p", "rb"))

