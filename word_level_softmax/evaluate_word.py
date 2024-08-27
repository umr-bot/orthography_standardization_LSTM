# coding: utf-8
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
from utils import WordTable
import numpy as np
from utils import transform2, datagen_simple
from word_model import lstm_model_n
from keras.models import load_model
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Baseline models training script")
parser.add_argument("--model_cnt", default=0, help="model to start or load from")
parser.add_argument("--data_dir",help="path to unigrams")
parser.add_argument("--checkpoints_dir", default="eng_checkpoints", help="directory in which to save checkpoints")
parser.add_argument("--foldset_num", default="1", help="foldset to load")

args = parser.parse_args()

args.model_cnt = int(args.model_cnt)

with open(args.data_dir+"/foldset"+args.foldset_num+"/train") as f: train = [line.strip('\n') for line in f]
with open(args.data_dir+"/norm_foldset"+args.foldset_num+"/train") as f: train_tar = [line.strip('\n') for line in f]
with open(args.data_dir+"/foldset"+args.foldset_num+"/val") as f: val = [line.strip('\n') for line in f]
with open(args.data_dir+"/norm_foldset"+args.foldset_num+"/val") as f: val_tar = [line.strip('\n') for line in f]

maxlen = max([len(tok) for tri in train_tar for tok in tri.split()])
tokens,dec_tokens = [tok for tri in train for tok in tri.split()],[tok for tri in train_tar for tok in tri.split()]
encoder_tokens, decoder_tokens, target_tokens = transform2(tokens=tokens,maxlen=maxlen,dec_tokens=
dec_tokens)
w_table = WordTable(tokens)
#b = batch(tokens=encoder_tokens, maxlen=maxlen, wtable=w_table, batch_size=10)
ind = 0
w_table.decode(w_table.encode(train[ind].split(), nb_rows=3))

#tri_encoder=[]
#for tri in train:
#    temp_encoder = []
#    for tok in tri.split():
#        encoder = tok
#        encoder += '*' * (maxlen - len(encoder)) # Padded to maxlen.
#        temp_encoder.append(encoder)
#    tri_encoder.append(temp_encoder)

def batch_tris(tris, maxlen, wtable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    def generate(tris, reverse):
        while(True): # This flag yields an infinite generator.
            for tri in tris:
                #if reverse:
                #    token = token[::-1]
                yield tri
    
    tri_iterator = generate(tris, reverse)
    data_batch = np.zeros((batch_size, 3, wtable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            tri = next(tri_iterator)
            data_batch[i] = wtable.encode(tri, 3)
            #except: print(token)
        yield data_batch

batch_size=10
train_tar_tris = [tri.split() for tri in train_tar]
train_decoder = batch_tris(tris=train_tar_tris, maxlen=3, wtable=w_table, batch_size=batch_size)
train_tris = [tri.split() for tri in train]
train_encoder = batch_tris(tris=train_tris, maxlen=3, wtable=w_table, batch_size=batch_size)

val_tar_tris = [tri.split() for tri in val_tar]
val_decoder = batch_tris(tris=val_tar_tris, maxlen=3, wtable=w_table, batch_size=batch_size)
val_tris = [tri.split() for tri in val]
val_encoder = batch_tris(tris=val_tris, maxlen=3, wtable=w_table, batch_size=batch_size)

#b1 = next(train_decoder)
#for batch_cnt in range(len(b1)): print(w_table.decode(b1[batch_cnt]))
if int(args.model_cnt) > 0:
    print(f"Loading model with epoch {str(args.model_cnt)}")
    model = load_model(args.checkpoints_dir+"/softmax_regression_"+str(args.model_cnt)+".h")

train_loader = datagen_simple(train_encoder, train_decoder)
val_loader = datagen_simple(val_encoder, val_decoder)

#for epoch in range(args.model_cnt,100):
#    print(f"Epoch: {str(epoch+1)} / 100")
y_decoded,y_pred_decoded,y_tar_decoded=[],[],[]
for batch in tqdm(range(0,len(val_tar_tris),batch_size), desc=f"Computing predictions"):
    y, y_tar = next(val_loader)
    y_pred = model.predict(y, verbose=0)
    for tri_cnt in range(len(y_pred)):
        y_decoded.append(w_table.decode(y[tri_cnt])[1])
        y_pred_decoded.append(w_table.decode(y_pred[tri_cnt])[1])
        y_tar_decoded.append(w_table.decode(y_tar[tri_cnt])[1])
        #print(y_decoded[tri_cnt][1],y_pred_decoded[tri_cnt][1],y_tar_decoded[tri_cnt][1])
save_dir = args.checkpoints_dir+"/model_"+str(args.model_cnt)
if not os.path.exists(save_dir): os.makedirs(save_dir)
with open(save_dir+"/err_file",'w') as f:
    for tri in y_decoded: f.write(tri+'\n')
with open(save_dir+"/cln_file",'w') as f:
    for tri in y_pred_decoded: f.write(tri+'\n')
with open(save_dir+"/tar_file",'w') as f:
    for tri in y_tar_decoded: f.write(tri+'\n')

