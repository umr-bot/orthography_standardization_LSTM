import re
import os
import unidecode
import numpy as np
from tqdm import tqdm
import json

np.random.seed(1234)

SOS = '\t' # start of sequence.
EOS = '*' # end of sequence.
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
BAM_CHARS = list("\-fsé\.lɲhaiyjuŋtɔ*opèxewbçknmvqcrɛzgd")
REMOVE_CHARS = '[#$%"\+@<=>!&,-.?:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'

def get_device():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device

class WordTable(object):
    """Given a set of words:
    + Encode them to a one-hot integer representation
    + Decode the one-hot integer representation to their word output
    + Decode a vector of probabilities to their word output
    """
    def __init__(self, words):
        """Initialize word table.
        # Arguments
          words: Words that can appear in the input, words === listofstrings
        """
        self.words = sorted(set(words+['<unk>']))
        self.word2index = dict((w, i) for i, w in enumerate(self.words))
        self.index2word = dict((i, w) for i, w in enumerate(self.words))
        #self.word2index['<unk>'] = max(self.index2word.keys())+1
        #self.index2word[max(self.index2word.keys())+1] = '<unk>'
        self.size = len(self.words)
    
    def encode(self, S, nb_rows):
        """One-hot encode given sentence S.
        # Arguments
          S: sentence, to be encoded.
          nb_rows: Number of rows in the returned one-hot encoding. This is
          used to keep the # of rows for each data the same via padding.
        """
        x = np.zeros((nb_rows, len(self.words)), dtype=np.float32)
        for i, w in enumerate(S):
            flag=True
            try: 
                x[i, self.word2index[w]] = 1.0
                flag=False
            except:
                #print(f"Word not working {S}")
                #x[i, 0] = 1.0
                #pass
                if flag: x[i, self.word2index['<unk>']] = 1.0
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their word output.
        # Arguments
          x: A vector or 2D array of probabilities or one-hot encodings,
          or a vector of word indices (used with `calc_argmax=False`).
          calc_argmax: Whether to find the word index with maximum
          probability, defaults to `True`.
        """
        if calc_argmax:
            indices = x.argmax(axis=-1)
        else:
            indices = x
        words = ' '.join(self.index2word[ind] for ind in indices)
        return indices, words

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
          chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char2index = dict((c, i) for i, c in enumerate(self.chars))
        self.index2char = dict((i, c) for i, c in enumerate(self.chars))
        self.size = len(self.chars)
    
    def encode(self, C, nb_rows):
        """One-hot encode given string C.
        # Arguments
          C: string, to be encoded.
          nb_rows: Number of rows in the returned one-hot encoding. This is
          used to keep the # of rows for each data the same via padding.
        """
        x = np.zeros((nb_rows, len(self.chars)), dtype=np.float32)
        for i, c in enumerate(C):
            try: x[i, self.char2index[c]] = 1.0
            except:
                #print(f"Char not working {C}")
                #x[i, 0] = 1.0
                continue
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
          x: A vector or 2D array of probabilities or one-hot encodings,
          or a vector of character indices (used with `calc_argmax=False`).
          calc_argmax: Whether to find the character index with maximum
          probability, defaults to `True`.
        """
        if calc_argmax:
            indices = x.argmax(axis=-1)
        else:
            indices = x
        chars = ''.join(self.index2char[ind] for ind in indices)
        return indices, chars

    def sample_multinomial(self, preds, temperature=1.0):
        """Sample index and character output from `preds`,
        an array of softmax probabilities with shape (1, 1, nb_chars).
        """
        # Reshaped to 1D array of shape (nb_chars,).
        preds = np.reshape(preds, len(self.chars)).astype(np.float64)
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        index = np.argmax(probs)
        char  = self.index2char[index]
        return index, char

def transform_3(tokens, maxlen):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    encoder_tokens = []
    for token in tokens:
        encoder = token
        encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
        encoder_tokens.append(encoder)
    
    return encoder_tokens

def transform2(tokens, maxlen, shuffle=False, dec_tokens=[], chrs=[], reverse=False):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []

    assert(len(tokens)==len(dec_tokens))
    for i in range(len(tokens)):
        token,dec_token = tokens[i], dec_tokens[i]
        if len(token) > 0: # only deal with tokens longer than length 3
            #encoder = add_speling_erors(token, error_rate=error_rate)
            encoder = token
            encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
            if reverse: encoder = encoder[::-1]
            #encoder_tokens.append(encoder)
        
            decoder = SOS + dec_token
            decoder += EOS * (maxlen - len(decoder))
            #decoder_tokens.append(decoder)
        
            target = decoder[1:]
            target += EOS * (maxlen - len(target))
            #target_tokens.append(target)
            if (len(encoder) == len(decoder) == len(target)):
                encoder_tokens.append(encoder)
                decoder_tokens.append(decoder)
                target_tokens.append(target)
            else: continue

    return encoder_tokens, decoder_tokens, target_tokens


def my_transform(tokens, maxlen, error_tokens, shuffle=False,reverse=False):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    if shuffle:
        print('Shuffling data.')
        np.random.shuffle(tokens)
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []
    for token in tokens:
        #encoder = add_speling_erors(token, error_rate=error_rate)
        encoder = error_tokens[token] # error_tokens is a dict mapping correct
                                      # tokens to possible errorful tokens
        encoder += EOS * (maxlen - len(encoder)) # Padded to maxlen.
        if reverse: encoder = encoder[::-1]
        encoder_tokens.append(encoder)

        decoder = SOS + token
        decoder += EOS * (maxlen - len(decoder))
        decoder_tokens.append(decoder)
    
        target = decoder[1:]
        target += EOS * (maxlen - len(target))
        target_tokens.append(target)
        assert(len(encoder) == len(decoder) == len(target))

    return encoder_tokens, decoder_tokens, target_tokens

def batch(tokens, maxlen, ctable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    def generate(tokens, reverse):
        while(True): # This flag yields an infinite generator.
            for token in tokens:
                if reverse:
                    token = token[::-1]
                yield token
    
    token_iterator = generate(tokens, reverse)
    data_batch = np.zeros((batch_size, maxlen, ctable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = ctable.encode(token, maxlen)
            #except: print(token)
        yield torch.from_numpy(data_batch)

def batch_triplet(token_triplets, maxlen, ctable, batch_size=128):
    def generate(token_triplets):
        while(True): # This flag yields an infinite generator.
            for token_triplet in token_triplets: yield token_triplet
    token_iterator = generate(token_triplets)
    data_batch = np.zeros((batch_size, 3, maxlen, ctable.size), dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = (ctable.encode(token[0], maxlen), ctable.encode(token[1], maxlen),ctable.encode(token[2], maxlen))
            #except: print(token)
        yield data_batch

def batch_bigram(token_pairs, maxlen, ctable, batch_size=128, reverse=False):
    """Split data into chunks of `batch_size` examples."""
    def generate(token_pairs, reverse):
        while(True): # This flag yields an infinite generator.
            for token_pair in token_pairs:
                #if reverse: token = token[::-1]
                yield token_pair
    
    token_iterator = generate(token_pairs, reverse)
    data_batch = np.zeros((batch_size, maxlen, ctable.size),
                          dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            data_batch[i] = ctable.encode(token, maxlen)
            #except: print(token)
        yield data_batch

def datagen_simple(input_iter, target_iter):
    """Utility function to load data into required model format."""
    while(True):
        input_ = next(input_iter)
        target = next(target_iter)
        yield (input_, target)

def datagen(encoder_iter, decoder_iter, target_iter):
    """Utility function to load data into required model format."""
    inputs = zip(encoder_iter, decoder_iter)
    while(True):
        encoder_input, decoder_input = next(inputs)
        target = next(target_iter)
        yield ([encoder_input, decoder_input], target)

def datagen_triplet(encoder_iter, decoder_iter, target_iter):
    """Utility function to load data into required model format."""
    inputs = zip(encoder_iter, decoder_iter)
    while(True):
        x0, decoder_input = next(inputs)
        x1, blank = next(inputs)
        x2, blank = next(inputs)
        target = next(target_iter)
        yield ([x0,x1,x2, decoder_input], target)

def pad_tris(tris, maxlen):
    tri_padded=[]
    for tri in tris:
        temp_padded = []
        for tok in tri.split():
            padded_tok = '#'+tok
            padded_tok += '*' * (maxlen - len(padded_tok)) # Padded to maxlen.
            temp_padded.append(padded_tok)
        tri_padded.append(temp_padded)
    return tri_padded

def batch_char_tri(tris,batch_size,ctable, maxlen):
    def generate(tris):
        while(True): # This flag yields an infinite generator.
            for tri in tris:
                yield tri

    tri_iterator = generate(tris)
    data_batch = np.zeros((batch_size, 3*maxlen, ctable.size), dtype=np.float32)
    while(True):
        for i in range(batch_size):
            tri = next(tri_iterator)
            a = ctable.encode(tri[0], nb_rows=maxlen)
            b = ctable.encode(tri[1], nb_rows=maxlen)
            c = ctable.encode(tri[2], nb_rows=maxlen)
            data_batch[i] = np.concatenate((a,b,c))
            #except: print(token)
        yield data_batch

