from utils import WordTable, transform2, datagen_simple, CharacterTable
import numpy as np
from word_model import select_model, softmax_regression

from tensorflow.keras.models import load_model
import argparse
import os
from tensorflow.keras.regularizers import L1L2

parser = argparse.ArgumentParser(description="Baseline models training script")
parser.add_argument("--num_epochs", default=1, help="number of epochs to train on")
parser.add_argument("--hidden_size", default=512, help="hidden dimension size")
parser.add_argument("--model_cnt", default=0, help="model to start or load from")
parser.add_argument("--train_batch_size", default=128, help="train batch size")
parser.add_argument("--val_batch_size", default=256, help="val batch size")
parser.add_argument("--checkpoints", default=0, help="dir to save models")
parser.add_argument("--data_dir", default=0, help="dir to data")
parser.add_argument("--num_layers", default=1, help="Number of layers to the model")
parser.add_argument("--model_name", default=0, help="Model name")
parser.add_argument("--embed_model_name", default='word2vec_padded.model', help="Model name for w2v embedding")
parser.add_argument("--embed_size", default=300, help="w2v embedding output size")
parser.add_argument("--bias_l1", default=0, help="bias l1 regularization")
parser.add_argument("--bias_l2", default=0, help="bias l2 regularization")
args = parser.parse_args()

args.model_cnt=int(args.model_cnt); args.train_batch_size=int(args.train_batch_size); args.val_batch_size=int(args.val_batch_size)
args.num_epochs=int(args.num_epochs);args.num_layers=int(args.num_layers);args.embed_size=int(args.embed_size)
args.bias_l1=float(args.bias_l1);args.bias_l2=float(args.bias_l2);
with open(args.data_dir + "/foldset1/train") as f: train = [line.strip('\n') for line in f]
with open(args.data_dir + "/norm_foldset1/train") as f: train_tar = [line.strip('\n') for line in f]
with open(args.data_dir + "/foldset1/val") as f: val = [line.strip('\n') for line in f]
with open(args.data_dir + "/norm_foldset1/val") as f: val_tar = [line.strip('\n') for line in f]

maxlen = max([len(tok) for tri in train_tar for tok in tri.split()])
tokens,dec_tokens = [tok for tri in train for tok in tri.split()],[tok for tri in train_tar for tok in tri.split()]
train_tokens,train_dec_tokens = [tok for tri in train for tok in tri.split()],[tok for tri in train_tar for tok in tri.split()]
val_tokens,val_dec_tokens = [tok for tri in val for tok in tri.split()],[tok for tri in val_tar for tok in tri.split()]

def pad_tris(tris):
    tri_padded=[]
    for tri in tris:
        temp_padded = []
        for tok in tri.split():
            padded_tok = '#'+tok
            padded_tok += '*' * (maxlen - len(padded_tok)) # Padded to maxlen.
            temp_padded.append(padded_tok)
        tri_padded.append(temp_padded)
    return tri_padded
train_padded, train_tar_padded = pad_tris(train), pad_tris(train_tar)
val_padded, val_tar_padded = pad_tris(val), pad_tris(val_tar)

def batch_char_tri(tris,batch_size,ctable):
    def generate(tris):
        while(True): # This flag yields an infinite generator.
            for tri in tris:
                yield tri

    tri_iterator = generate(tris)
    data_batch = np.zeros((batch_size, 3*maxlen, ctable.size), dtype=np.float32)
    while(True):
        for i in range(batch_size):
            tri = next(tri_iterator)
            a = ctable.encode(tri[0], nb_rows=maxlen);b = ctable.encode(tri[1], nb_rows=maxlen);c = ctable.encode(tri[2], nb_rows=maxlen)
            data_batch[i] = np.concatenate((a,b,c))
            #except: print(token)
        yield data_batch

def batch_char_tri(tris,batch_size,ctable):
    def generate(tris):
        while(True): # This flag yields an infinite generator.
            for tri in tris:
                yield tri

    tri_iterator = generate(tris)
    data_batch = np.zeros((batch_size, 3*maxlen), dtype=np.float32)
    while(True):
        for i in range(batch_size):
            tri = next(tri_iterator)
            a = ctable.decode(ctable.encode(tri[0], nb_rows=maxlen))[0]
            b = ctable.decode(ctable.encode(tri[1], nb_rows=maxlen))[0]
            c = ctable.decode(ctable.encode(tri[2], nb_rows=maxlen))[0]
            data_batch[i] = np.concatenate((a,b,c))
            #except: print(token)
        yield data_batch

##w_table = WordTable(train_tokens)
##def batch_word_tris(tris, maxlen, wtable, batch_size=128, reverse=False):
##    """Split data into chunks of `batch_size` examples."""
##    def generate(tris, reverse):
##        while(True): # This flag yields an infinite generator.
##            for tri in tris:
##                #if reverse:
##                #    token = token[::-1]
##                yield tri
##    
##    tri_iterator = generate(tris, reverse)
##    data_batch = np.zeros((batch_size, ctable.size))
##    while(True):
##        for i in range(batch_size):
##            tri = next(tri_iterator)
##            data_batch[i] = w_table.decode(wtable.encode(tri, 3))[0][1]
##            #except: print(token)
##        yield data_batch
#
##train_tar_tris = [tri.split() for tri in train_tar]
##train_decoder = batch_word_tris(tris=train_tar_tris, maxlen=3, wtable=w_table, batch_size=args.train_batch_size)
##val_tar_tris = [tri.split() for tri in val_tar]
##val_decoder = batch_word_tris(tris=val_tar_tris, maxlen=3, wtable=w_table, batch_size=args.val_batch_size)

chars = set(c for tok in train_tokens for c in tok).union( set(('#','*')) )
ctable =  CharacterTable(chars)

train_encoder = batch_char_tri(tris=train_padded, batch_size=args.train_batch_size,ctable=ctable)
train_decoder = batch_char_tri(tris=train_tar_padded, batch_size=args.train_batch_size,ctable=ctable)
val_encoder = batch_char_tri(tris=val_padded, batch_size=args.val_batch_size,ctable=ctable)
val_decoder = batch_char_tri(tris=val_tar_padded, batch_size=args.val_batch_size,ctable=ctable)

if int(args.model_cnt) > 0:
    print(f"Loading model with epoch {str(args.model_cnt)}")
    model = load_model(args.checkpoints+"/word_model_"+str(args.model_cnt)+".h")
else:
    if not os.path.exists(args.checkpoints): os.makedirs(args.checkpoints)
    with open(args.checkpoints+'/history','w') as f: f.write("") # empty file for history values if model=0
    #model = lstm_model_n(hidden_size=args.hidden_size, nb_features=ctable.size, n_layers=args.num_layers, bias_reg=L1L2(l1=args.bias_l1,l2=args.bias_l2))
    #model = select_model(hidden_size=512, nb_features=w_table.size, n_layers=args.num_layers, model_name=args.model_name, maxlen=3, embed_size=args.embed_size)
    model = softmax_regression(nb_features=3*maxlen)
train_loader = datagen_simple(train_encoder, train_decoder)
val_loader = datagen_simple(val_encoder, val_decoder)

train_steps = len(train_tokens) // (args.train_batch_size)
val_steps = len(val_tokens) // (args.val_batch_size)
print("Number of train_steps:",train_steps)
print("Number of val_steps:",val_steps)

for epoch in range(args.model_cnt,args.num_epochs+1):
    print(f"Epoch: {str(epoch+1)} / {str(args.num_epochs)}")
    #history = model.fit(train_loader, steps_per_epoch=100, , epochs=1, verbose=1)
    history = model.fit(train_loader, steps_per_epoch=train_steps, validation_data=val_loader, validation_steps=val_steps, epochs=1, verbose=1)

    #if not os.path.exists(args.checkpoints): os.makedirs(args.checkpoints)
    if epoch%10==0: 
        print(f"Saving model to {args.checkpoints}/{args.model_name}_{str(epoch)}.h")
        model.save(args.checkpoints+"/"+args.model_name+"_"+str(epoch)+".h")
    if epoch%10==0:
        y, y_tar = next(val_loader)
        y_pred = model.predict(y)
        y_decoded,y_pred_decoded,y_tar_decoded=[],[],[]
        for tri_cnt in range(3):
            #y_decoded.append(ctable.decode(y[tri_cnt]))
            #y_pred_decoded.append(ctable.decode(y_pred[tri_cnt]))
            #y_tar_decoded.append(ctable.decode(y_tar[tri_cnt]))
            y_decoded.append(ctable.decode(y[tri_cnt].astype(int), calc_argmax=False))
            y_pred_decoded.append(ctable.decode(y_pred[tri_cnt].astype(int), calc_argmax=False))
            y_tar_decoded.append(ctable.decode(y_tar[tri_cnt].astype(int), calc_argmax=False))

            print(y_decoded[tri_cnt][1],y_pred_decoded[tri_cnt][1],y_tar_decoded[tri_cnt][1])

    with open(args.checkpoints+'/history','a') as f:
        f.write(str(history.history['loss'][0]) + ',')
        f.write(str(history.history['val_loss'][0]) + ',')
        f.write(str(history.history['accuracy'][0]) + ',')
        f.write(str(history.history['val_accuracy'][0]))
        f.write('\n')