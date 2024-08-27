# StackOverflow question: https://stackoverflow.com/questions/51123481/how-to-build-a-language-model-using-lstm-that-assigns-probability-of-occurence-f
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import Sequence
from keras.models import Sequential
import keras
import numpy as np


def prepare_sentence(seq, maxlen):
    # Pads seq and slides windows
    x = []
    y = []
    for i, w in enumerate(seq):
        x_padded = pad_sequences([seq[:i]],
                                 maxlen=maxlen - 1,
                                 padding='pre')[0]  # Pads before each sequence
        x.append(x_padded)
        y.append(w)
    return x, y


class DataGenerator(Sequence):
    # Memory efficient data generator for feeding sentences
    def __init__(self, sentences, batch_size=32, shuffle=True):
        # Initilise
        self.sentences = sentences
        self.indexes = np.arange(len(sentences))
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Preprocess data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data)
        self.tokenizer = tokenizer
        self.vocab = tokenizer.word_index
        self.seqs = tokenizer.texts_to_sequences(data)
        self.maxlen = max([len(seq) for seq in self.seqs])

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.sentences) / self.batch_size))

    def __getitem__(self, index):
        # Load one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Get seqhences for the batch
        seqs = [self.seqs[k] for k in indexes]

        # Slide windows and pad selected sequences
        x = []
        y = []
        for seq in seqs:
            x_windows, y_windows = prepare_sentence(seq, self.maxlen)
            x += x_windows
            y += y_windows
        x = np.array(x)
        y = np.array(y) - 1
        y = np.eye(len(self.vocab))[y]  # One hot encoding

        return x, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    # Data
    data = ["Two little dicky birds",
            "Sat on a wall,",
            "One called Peter,",
            "One called Paul.",
            "Fly away, Peter,",
            "Fly away, Paul!",
            "Come back, Peter,",
            "Come back, Paul."]

    # Preprocess data
    generator = DataGenerator(sentences=data,
                              batch_size=4,
                              shuffle=True)

    # Define model
    model = Sequential()
    vocab_size = len(generator.vocab)
    model.add(Embedding(input_dim=vocab_size + 1,  # vocabulary size. Adding an
                        # extra element for <PAD> word
                        output_dim=5,  # size of embeddings
                        input_length=generator.maxlen - 1))  # length of the padded sequences
    model.add(LSTM(10))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile('rmsprop', 'categorical_crossentropy')

    # Train network
    model.fit_generator(generator, epochs=1000, verbose=1)

    # Compute probability of occurence of a sentence
    sentence = data[0]
    tok = generator.tokenizer.texts_to_sequences([sentence])[0]
    x_test, y_test = prepare_sentence(tok, generator.maxlen)
    x_test = np.array(x_test)
    y_test = np.array(y_test) - 1  # The word <PAD> does not have a class
    p_pred = model.predict(x_test)
    vocab_inv = {v: k for k, v in generator.vocab.items()}
    log_p_sentence = 0
    for i, prob in enumerate(p_pred):
        word = vocab_inv[y_test[i] + 1]  # Index 0 from vocab is reserved to <PAD>
        history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
        prob_word = prob[y_test[i]]
        log_p_sentence += np.log(prob_word)
        print('P(w={}|h={})={}'.format(word, history, prob_word))
    print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
