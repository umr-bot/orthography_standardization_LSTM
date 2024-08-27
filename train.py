# coding: utf-8
import os
from utils import CharacterTable, transform2, batch, datagen_simple, transform_3
from utils_2 import train_generator, train_generator_2
from models import lstm_model, lstm_model_n, softmax_regression,softmax_regression_2, lstm_model_n_dropout, bi_lstm_model_n, lstm_embed
import pickle
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
#from keras.optimizers import RMSprop
from model import precision,recall,f1_score

import argparse

parser = argparse.ArgumentParser(description="Baseline models training script")
parser.add_argument("--hidden_size", default=128, help="hidden layer1 size")
parser.add_argument("--dropout", default=0.2, help="dropout regularization rate")
parser.add_argument("--num_epochs", help="hidden layer1 size")
parser.add_argument("--data_dir",help="path to unigrams")
#parser.add_argument("--lang",help="language being trained on")
parser.add_argument("--train_batch_size", default=128, help="train batch size")
parser.add_argument("--val_batch_size", default=256, help="val batch size")
parser.add_argument("--foldset_num", default=1, help="foldset number to use")
parser.add_argument("--checkpoints_dir", default="eng_checkpoints", help="directory in which to save checkpoints")
parser.add_argument("--history_fn", default="history.txt", help="file name to save model history in")
parser.add_argument("--model", help="model type to train with")
parser.add_argument("--model_num", default=0, help="Model to start training from")
parser.add_argument("--use_bigrams", default="false", help="boolean which selects whether to load unigram or bigram data")
parser.add_argument("--use_trigrams", default="false", help="boolean which selects whether to load trigram data")
parser.add_argument("--multilang_dir", default=None, help="boolean which selects whether to load multilingual unigram data")
args = parser.parse_args()
args.num_epochs, args.train_batch_size, args.val_batch_size, args.dropout = int(args.num_epochs), int(args.train_batch_size), int(args.val_batch_size), float(args.dropout)

if args.multilang_dir != None:
    train_dec_tokens, train_tokens, val_dec_tokens, val_tokens = [],[],[],[]
    with open(args.multilang_dir) as f:
        for line in f:
            for tok in line.split():
                if len(tok)>2: train_dec_tokens.append(tok)
    train_tokens = train_dec_tokens.copy(); val_dec_tokens = train_dec_tokens.copy(); val_tokens = train_dec_tokens.copy()
    interval = 10000
    train_tokens = train_tokens[:interval]; train_dec_tokens = train_dec_tokens[0:interval]
    val_tokens = val_tokens[interval:int(1.2*interval)]; val_dec_tokens = val_dec_tokens[interval:int(1.2*interval)]
elif args.use_bigrams == "right" or args.use_bigrams == "left":
    # selects bigram (n=2) or trigram (n=3)
    if args.use_bigrams == "left": m,n=0,2 # left bigram (m=0,n=2)
    else: m,n=-2,3 # right bigram (m=-2,n=3)
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
        train_tokens = [" ".join(line.strip('\n').split()[m:n]) for line in f]
    with open(args.data_dir+"/norm_foldset"+str(args.foldset_num)+"/train") as f:
        train_dec_tokens = [" ".join(line.strip('\n').split()[m:n]) for line in f]

    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
        val_tokens = [" ".join(line.strip('\n').split()[m:n]) for line in f]
    with open(args.data_dir+"/norm_foldset"+str(args.foldset_num)+"/val") as f:
        val_dec_tokens = [" ".join(line.strip('\n').split()[m:n]) for line in f]
elif args.use_trigrams == "true":
    # selects bigram (n=2) or trigram (n=3)
    n=3
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
        train_tokens = [" ".join(line.strip('\n').split()[:n]) for line in f]
    with open(args.data_dir+"/norm_foldset"+str(args.foldset_num)+"/train") as f:
        train_dec_tokens = [" ".join(line.strip('\n').split()[:n]) for line in f]

    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
        val_tokens = [" ".join(line.strip('\n').split()[:n]) for line in f]
    with open(args.data_dir+"/norm_foldset"+str(args.foldset_num)+"/val") as f:
        val_dec_tokens = [" ".join(line.strip('\n').split()[:n]) for line in f]

else: print("ERROR: data not loaded")

input_chars = set(' '.join(train_tokens) + '*' + '\t') # * and \t are EOS and SOS respectively
target_chars = set(' '.join(train_dec_tokens) + '*' + '\t')
total_chars = input_chars.union(target_chars)
nb_total_chars = len(total_chars)
total_ctable = input_ctable = target_ctable = CharacterTable(total_chars)
maxlen = max([len(token) for token in train_tokens]) + 2

train_steps = len(train_tokens) // args.num_epochs
val_steps = len(val_tokens) // args.num_epochs

if args.multilang_dir != None and int(args.model_num) > 0:
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
        train_tups = [line.strip('\n').split(',') for line in f]
    train_dec_tokens, train_tokens = zip(*train_tups)

    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
        val_tups = [line.strip('\n').split(',') for line in f]
    val_dec_tokens, val_tokens = zip(*val_tups)

# If a model number is given as a argument that is non zero to load from
if int(args.model_num) > 0: 
    custom_objects = { 'recall': recall, "precision": precision, "f1_score": f1_score}
    root_dir=args.checkpoints_dir+'/fold' + str(args.foldset_num)+'_hdd_'+str(args.hidden_size)
    model = load_model( root_dir + "/seq2seq_epoch_"+args.model_num+".h5",custom_objects=custom_objects)

elif args.model=='lstm_1': model = lstm_model_n(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=1)
elif args.model=='lstm_2': model = lstm_model_n(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=2)
elif args.model=='lstm_3': model = lstm_model_n(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=3)
elif args.model=='lstm_1_dropout': model = lstm_model_n_dropout(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=1,dropout=args.dropout)
elif args.model=='lstm_2_dropout': model = lstm_model_n_dropout(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=2,dropout=args.dropout)
elif args.model=='lstm_3_dropout': model = lstm_model_n_dropout(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=3,dropout=args.dropout)
elif args.model=='bi_lstm_1': model = bi_lstm_model_n(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=1)
elif args.model=='bi_lstm_2': model = bi_lstm_model_n(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=2)
elif args.model=='bi_lstm_3': model = bi_lstm_model_n(hidden_size=int(args.hidden_size),nb_features=total_ctable.size,n_layers=3)
elif args.model=='softmax_regression': model = softmax_regression(hidden_size=int(args.hidden_size),nb_features=total_ctable.size)
elif args.model=='softmax_regression_2': model = softmax_regression_2(hidden_size=int(args.hidden_size),nb_features=total_ctable.size)
elif args.model=='lstm_embed': model = lstm_embed(hidden_size=int(args.hidden_size), nb_features=total_ctable.size, n_layers=2, embed_output_dim=int(0.2*total_ctable.size), maxlen=maxlen)
#elif args.model=='lstm_2_embed': model = lstm_2_embed(nb_features=total_ctable.size, embed_output_dim=2*total_ctable.size, maxlen=maxlen, opt='adam')
###train_batch_size=round(len(train_tokens),-3)//100
###val_batch_size=round(len(val_tokens),-3)//50

start=int(args.model_num)
for epoch in range(start,args.num_epochs):
    print(f"Epoch {str(epoch+1)}")
    st_ind,et_ind = int(len(train_tokens)*(epoch/args.num_epochs)), int(len(train_tokens)*((epoch+1)/args.num_epochs) )
    sv_ind,ev_ind = int(len(val_tokens)*(epoch/args.num_epochs)), int(len(val_tokens)*((epoch+1)/args.num_epochs))

    train_x_padded=transform_3(train_tokens[st_ind:et_ind], maxlen=maxlen)
    train_y_padded=transform_3(train_dec_tokens[st_ind:et_ind], maxlen=maxlen)

    val_x_padded=transform_3(val_tokens[sv_ind:ev_ind], maxlen=maxlen)
    val_y_padded=transform_3(val_dec_tokens[sv_ind:ev_ind], maxlen=maxlen)

    train_X_iter = batch(train_x_padded,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size,reverse=False)
    train_y_iter = batch(train_y_padded,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size,reverse=False)
    val_X_iter = batch(val_x_padded,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size,reverse=False)
    val_y_iter = batch(val_y_padded,maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size,reverse=False)

    train_loader = datagen_simple(train_X_iter, train_y_iter)
    val_loader = datagen_simple(val_X_iter, val_y_iter)

    #tg = train_generator_2(train_tokens[st_ind:et_ind],train_dec_tokens[st_ind:et_ind],maxlen=maxlen,ctable=total_ctable,batch_size=args.train_batch_size)
    #vg = train_generator_2(val_tokens[sv_ind:ev_ind],val_dec_tokens[sv_ind:ev_ind],maxlen=maxlen,ctable=total_ctable,batch_size=args.val_batch_size)

    history = model.fit(train_loader, steps_per_epoch=train_steps, validation_data=val_loader, validation_steps=val_steps, epochs=1, verbose=1)

    for batch_cnt in range(3):
        val_batch = next(val_loader)
        # Get one sample from batch of validation data
        x,y=val_batch[0],val_batch[1]
        # model.predict computes predictions for batches of samples, batch_size is min at 1 sample
        y_pred = model.predict(x)
        batch_size = 1
        # uncomment line below to loop over samples if using batch_size > 1 and change y_pred[0] to y_pred[word_ind]
        for word_ind in range(batch_size):
            val_x = total_ctable.decode(x[word_ind], calc_argmax=True)[1]
            val_y_pred = total_ctable.decode(y_pred[word_ind], calc_argmax=True)[1]
            val_y_true = total_ctable.decode(y[word_ind], calc_argmax=True)[1]
            print(val_x,val_y_pred,val_y_true)

    # Save the model at end of each epoch.
    model_file = '_'.join(['seq2seq', 'epoch', str(epoch + 1)]) + '.h5'
    save_dir=args.checkpoints_dir + '/fold' + str(args.foldset_num)+'_hdd_'+args.hidden_size

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

