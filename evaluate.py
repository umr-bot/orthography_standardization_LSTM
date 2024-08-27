# coding: utf-8
import os
# NOTE: log levels for tensorflow
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model, load_model
from feature_analyzer import get_iter
from tqdm import tqdm
from model import precision,recall,f1_score
from utils import CharacterTable, batch, datagen_simple, transform_3
import argparse

parser = argparse.ArgumentParser(description="Logistic regression training script")
parser.add_argument("--hidden_size", default=128, help="hidden layer1 size")
parser.add_argument("--train_batch_size", default=10, help="train batch size")
parser.add_argument("--val_batch_size", default=100, help="val batch size")
parser.add_argument("--num_epochs", default=20, help="number of epochs")
parser.add_argument("--foldset_num", default=1, help="foldset number to use")
parser.add_argument("--checkpoints_dir", default="eng_checkpoints", help="directory in which to save checkpoints")
parser.add_argument("--data_dir",help="path to unigrams")
#parser.add_argument("--lang",help="language being trained on")
parser.add_argument("--model_num",help="which model version is being evaluated on")
parser.add_argument("--batch_size", default=128, help="size of batches to use in inference")
parser.add_argument("--use_bigrams", default="false", help="boolean which selects whether to load unigram or bigram data")
parser.add_argument("--use_trigrams", default="false", help="boolean which selects whether to load unigram or trigram data")
parser.add_argument("--multilang_dir", default=None, help="boolean which selects whether to load multilingual unigram data")

args = parser.parse_args()
args.hidden_size, args.num_epochs, args.train_batch_size,args.val_batch_size=int(args.hidden_size),int(args.num_epochs),int(args.train_batch_size), int(args.val_batch_size)
#assert(len(int(args.val_batch_size)) < len(int(args.)))

if args.multilang_dir != None:
    train_dec_tokens, train_tokens, val_dec_tokens, val_tokens = [],[],[],[]
    with open(args.multilang_dir) as f:
        for line in f:
            for tok in line.split():
                if len(tok) > 2: train_dec_tokens.append(tok)
    train_tokens = train_dec_tokens.copy(); val_dec_tokens = train_dec_tokens.copy(); val_tokens = train_dec_tokens.copy()
    interval = 10000
    train_tokens = train_tokens[:interval]; train_dec_tokens = train_dec_tokens[0:interval]
    val_tokens = val_tokens[interval:int(1.2*interval)]; val_dec_tokens = val_dec_tokens[interval:int(1.2*interval)]

elif args.use_bigrams == "false" and args.use_trigrams == "false":
    #train_dec_tokens are targets and train_tokens inputs
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
        train_tups = [line.strip('\n').split(',') for line in f]
    train_dec_tokens, train_tokens = zip(*train_tups)

    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
        val_tups = [line.strip('\n').split(',') for line in f]
    val_dec_tokens, val_tokens = zip(*val_tups)
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
#########################################################################################

if args.multilang_dir != None:
    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/train") as f:
        train_tups = [line.strip('\n').split(',') for line in f]
    train_dec_tokens, train_tokens = zip(*train_tups)

    with open(args.data_dir+"/foldset"+str(args.foldset_num)+"/val") as f:
        val_tups = [line.strip('\n').split(',') for line in f]
    val_dec_tokens, val_tokens = zip(*val_tups)


# Load model
custom_objects = { 'recall': recall, "precision": precision, "f1_score": f1_score}
root_dir=args.checkpoints_dir+'/fold' + str(args.foldset_num)+'_hdd_'+str(args.hidden_size)
model = load_model( root_dir + "/seq2seq_epoch_"+args.model_num+".h5",custom_objects=custom_objects)
#########################################################################################

val_x, val_y_pred,val_y_true=[],[],[]
#for epoch in tqdm(range(args.num_epochs)):
#train_iter, val_iter, total_ctable = get_iter(epoch=epoch,num_epochs=args.num_epochs ,fold_num=args.foldset_num,train_batch_size=int(args.train_batch_size),val_batch_size=int(args.val_batch_size), predict=True, data_dir=args.data_dir)

#sv_ind,ev_ind = int(args.val_batch_size*(epoch/args.num_epochs)), int(args.val_batch_size*((epoch+1)/args.num_epochs))
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
    y_pred = model.predict(x)
    # uncomment line below to loop over samples if using batch_size > 1 and change y_pred[0] to y_pred[word_ind]
    for word_ind in range(batch_size):
        val_x.append(total_ctable.decode(x[word_ind], calc_argmax=True)[1])
        val_y_pred.append(total_ctable.decode(y_pred[word_ind], calc_argmax=True)[1])
        val_y_true.append(total_ctable.decode(y[word_ind], calc_argmax=True)[1])

save_dir = root_dir+"/model_"+args.model_num
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+"/err_file",'w') as f:
    for word in val_x: f.write(word+'\n')
with open(save_dir+"/cln_file",'w')as f:
    for word in val_y_pred: f.write(word+'\n')
with open(save_dir+"/tar_file",'w') as f:
    for word in val_y_true: f.write(word+'\n')

