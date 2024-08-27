# coding: utf-8
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
from utils import WordTable, CharacterTable, batch_char_tri, pad_tris
import numpy as np
from utils import transform2, datagen_simple
#from models import lstm_model_n
from tensorflow.keras.models import load_model
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tqdm import tqdm

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
args = parser.parse_args()

args.model_cnt=int(args.model_cnt); args.train_batch_size=int(args.train_batch_size); args.val_batch_size=int(args.val_batch_size)
args.num_epochs=int(args.num_epochs);args.num_layers=int(args.num_layers);args.embed_size=int(args.embed_size)

with open(args.data_dir + "/foldset1/train") as f: train = [line.strip('\n') for line in f]
with open(args.data_dir + "/norm_foldset1/train") as f: train_tar = [line.strip('\n') for line in f]
with open(args.data_dir + "/foldset1/val") as f: val = [line.strip('\n') for line in f]
with open(args.data_dir + "/norm_foldset1/val") as f: val_tar = [line.strip('\n') for line in f]

maxlen = max([len(tok) for tri in train_tar for tok in tri.split()])
tokens,dec_tokens = [tok for tri in train for tok in tri.split()],[tok for tri in train_tar for tok in tri.split()]
train_tokens,train_dec_tokens = [tok for tri in train for tok in tri.split()],[tok for tri in train_tar for tok in tri.split()]
val_tokens,val_dec_tokens = [tok for tri in val for tok in tri.split()],[tok for tri in val_tar for tok in tri.split()]

train_padded, train_tar_padded = pad_tris(train, maxlen), pad_tris(train_tar, maxlen)
val_padded, val_tar_padded = pad_tris(val, maxlen), pad_tris(val_tar, maxlen)

chars = set(c for tok in train_tokens for c in tok).union( set(('#','*')) )
ctable =  CharacterTable(chars)

train_encoder = batch_char_tri(tris=train_padded, batch_size=args.train_batch_size,ctable=ctable, maxlen=maxlen)
train_decoder = batch_char_tri(tris=train_tar_padded, batch_size=args.train_batch_size,ctable=ctable, maxlen=maxlen)
val_encoder = batch_char_tri(tris=val_padded, batch_size=args.val_batch_size,ctable=ctable, maxlen=maxlen)
val_decoder = batch_char_tri(tris=val_tar_padded, batch_size=args.val_batch_size,ctable=ctable, maxlen=maxlen)

train_loader = datagen_simple(train_encoder, train_decoder)
val_loader = datagen_simple(val_encoder, val_decoder)

print(f"Loading model with epoch {str(args.model_cnt)}")
model = load_model(args.checkpoints+"/"+args.model_name+"_"+str(args.model_cnt)+".h")

#for epoch in range(args.model_cnt,100):
#    print(f"Epoch: {str(epoch+1)} / 100")
y_decoded,y_pred_decoded,y_tar_decoded=[],[],[]
for batch in tqdm(range(len(val_tar)), desc=f"Computing predictions"):
        y, y_tar = next(val_loader)
        y_pred = model.predict(y,verbose=0)
        for tri_cnt in range(3):
            y_decoded.append(ctable.decode(y[tri_cnt]))
            y_pred_decoded.append(ctable.decode(y_pred[tri_cnt]))
            y_tar_decoded.append(ctable.decode(y_tar[tri_cnt]))
            #print(y_decoded[tri_cnt][1],y_pred_decoded[tri_cnt][1],y_tar_decoded[tri_cnt][1])

save_dir = args.checkpoints+"/model_"+str(args.model_cnt)
if not os.path.exists(save_dir): os.makedirs(save_dir)
with open(save_dir+"/err_file",'w') as f:
    for tri in y_decoded: f.write(" ".join(tri[1].split('#'))+'\n')
with open(save_dir+"/cln_file",'w') as f:
    for tri in y_pred_decoded: f.write(" ".join(tri[1].split('#'))+'\n')
with open(save_dir+"/tar_file",'w') as f:
    for tri in y_tar_decoded: f.write(" ".join(tri[1].split('#'))+'\n')

