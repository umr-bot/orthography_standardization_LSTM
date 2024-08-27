#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import Sequence
import numpy as np

#def prepare_sentence(seq, maxlen):
#    # Pads seq and slides windows
#    x = []
#    y = []
#    for i, w in enumerate(seq):
#        x_padded = pad_sequences([seq[i:]],
#                                 maxlen=maxlen - 1,
#                                 padding='post')[0]  # Pads before each sequence
#        x.append(x_padded)
#        y.append(w)
#    return x, y
#
#class DataGenerator(Sequence):
#    # Memory efficient data generator for feeding sentences
#    def __init__(self, sentences, batch_size=32, shuffle=True):
#        # Initilise
#        self.sentences = sentences
#        self.indexes = np.arange(len(sentences))
#        self.batch_size = batch_size
#        self.shuffle = shuffle
#
#        # Preprocess data
#        tokenizer = Tokenizer()
#        tokenizer.fit_on_texts(sentences)
#        self.tokenizer = tokenizer
#        self.vocab = tokenizer.word_index
#        self.seqs = tokenizer.texts_to_sequences(sentences)
#        self.maxlen = max([len(seq) for seq in self.seqs])
#
#        self.on_epoch_end()
#
#    def __len__(self):
#        # Denotes the number of batches per epoch
#        return int(np.floor(len(self.sentences) / self.batch_size))
#
#    def __getitem__(self, index):
#        # Load one batch of data
#        # Generate indexes of the batch
#        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
#
#        # Get seqhences for the batch
#        seqs = [self.seqs[k] for k in indexes]
#
#        # Slide windows and pad selected sequences
#        x = []
#        y = []
#        for seq in seqs:
#            x_windows, y_windows = prepare_sentence(seq, self.maxlen)
#            x += x_windows
#            y += y_windows
#        x = np.array(x)
#        #x = np.eye(len(self.vocab))[x]
#        y = np.array(y) - 1
#        y = np.eye(len(self.vocab))[y]  # One hot encoding
#
#        return x, y
#
#    def on_epoch_end(self):
#        # Updates indexes after each epoch
#        if self.shuffle:
#            np.random.shuffle(self.indexes)

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

