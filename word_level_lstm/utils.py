#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import Sequence
import numpy as np
SOS = '\t' # start of sequence.
EOS = '*' # end of sequence.

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

def transform2(tokens, maxlen, shuffle=False, dec_tokens=[], chrs=[], reverse=False):
    """Transform tokens into model inputs and targets.
    All inputs and targets are padded to maxlen with EOS character.
    """
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []
    copy_tokens,copy_dec_tokens=[],[]
    if chrs != []:
        for i in range(len(tokens)):
            tok,dec_tok = tokens[i], dec_tokens[i]
            if set(tok).issubset(chrs) and set(dec_tok).issubset(chrs):
                copy_tokens.append(tok)
                copy_dec_tokens.append(dec_tok)
        tokens, dec_tokens = copy_tokens,copy_dec_tokens

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

def datagen_simple(input_iter, target_iter):
    """Utility function to load data into required model format."""
    while(True):
        input_ = next(input_iter)
        target = next(target_iter)
        yield (input_, target)

