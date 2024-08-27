import numpy as np
from utils import CharacterTable, transform2

def train_generator(input_tokens,target_tokens, maxlen, ctable, batch_size=10):
    #input_tokens=["one**","two**","three"]
    #target_tokens=["onne*","two**","thre*"]
    #maxlen,batch_size,ctable,ctable_size=5,1,total_ctable,total_ctable.size
    def generate(tokens):
        while(True): # This flag yields an infinite generator.
            for token in tokens:
                yield token
    input_token_iterator = generate(input_tokens)
    x_batch = np.zeros((batch_size*maxlen, maxlen, ctable.size),dtype=np.float32)
    target_token_iterator = generate(target_tokens)
    y_batch = np.zeros((batch_size*maxlen, maxlen, ctable.size),dtype=np.float32)
    #print(x_batch.shape,y_batch.shape)
    while True:
        for batch_num in range(0,batch_size*(maxlen-1)-len(input_tokens)):
            x_token,y_token = next(input_token_iterator), next(target_token_iterator)
            #tok_len = max(len(x_token), len(y_token))
            for i in range(maxlen-1):
                #x_token,y_token = next(input_token_iterator), next(target_token_iterator)
                #print(i, y_token)
                x_batch[batch_num+i] = ctable.encode(x_token[:i], maxlen)
                y_batch[batch_num+i] = ctable.encode(y_token[i], maxlen)
        yield x_batch, y_batch 

def train_generator_2(input_tokens,target_tokens, maxlen, ctable, batch_size=10):
    #input_tokens=train_tokens[0:20]
    #target_tokens=train_dec_tokens[0:20]
    #maxlen,batch_size,ctable,ctable_size=maxlen,3,total_ctable,total_ctable.size
    def generate(tokens):
        while(True): # This flag yields an infinite generator.
            for token in tokens:
                yield token
    max_input_tok_len=max([len(item) for item in input_tokens])
    input_token_iterator = generate(input_tokens)
    x_batch = np.zeros((batch_size*max_input_tok_len, maxlen, ctable.size),dtype=np.float32)
    target_token_iterator = generate(target_tokens)
    y_batch = np.zeros((batch_size*max_input_tok_len, maxlen, ctable.size),dtype=np.float32)
    while True:
        #print(x_batch.shape,y_batch.shape)
        #for batch_num in range(0,batch_size):
        batch_num=0
        while batch_num < (batch_size*max_input_tok_len)-max_input_tok_len:
            x_token,y_token = next(input_token_iterator), next(target_token_iterator)
            #print(x_token,y_token)
            for i in range(0,len(y_token)):
                #x_token,y_token = next(input_token_iterator), next(target_token_iterator)
                #if len(y_token) >= len(x_token):
                x_batch[batch_num+i] = ctable.encode(y_token[:i], nb_rows=maxlen)
                temp_tok = '\t'*i + y_token[i] + '*'*len(y_token[i:])
                y_batch[batch_num+i] = ctable.encode(temp_tok, nb_rows=maxlen)
                #print(batch_num+i)
            batch_num+=len(y_token)
        del_inds=[]
        for i in range(x_batch.shape[0]):
            if set(ctable.decode(x_batch[i], calc_argmax=True)[0]) == {0} or set(ctable.decode(y_batch[i], calc_argmax=True)[0]) == {0} : del_inds.append(i)
        x,y=np.delete(x_batch,del_inds,axis=0),np.delete(y_batch,del_inds,axis=0)
        yield x,y

###def datagen(zipped_IO):
###    """Utility function to load data into required model format."""
###    while(True):
###        Input, target = next(zipped_IO)
###        yield (Input, target)

