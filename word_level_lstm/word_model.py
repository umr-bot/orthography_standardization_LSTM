##from tensorflow.keras.layers import Embedding, LSTM, Dense, Reshape, RNN, LSTMCell
##from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, Reshape, RNN, LSTMCell, InputLayer, Bidirectional
#from keras.models import Sequential
from tensorflow.keras import Sequential, Input

import keras
import numpy as np
import tensorflow as tf

#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
#tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

def select_model(hidden_size, nb_features, n_layers, model_name, maxlen, embed_size=300):
    model=None
    if model_name == "softmax_regression": model = softmax_regression(nb_features)
    elif model_name == "lstm_word_n": model = lstm_model_n(hidden_size, nb_features, n_layers)
    elif model_name == "lstm_word_n_embed": model = lstm_model_n_embed(hidden_size, nb_features, n_layers, maxlen, embed_size, w2v_model_name="./word2vec_padded.model")
    elif model_name == "bi_lstm_word_n": model = bi_lstm_model_n(hidden_size, nb_features, n_layers)
    else: print("ERROR: Not a valid model name.")
    return model

def softmax_regression(nb_features):
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():

    model = Sequential()
    model.add(Dense(nb_features, activation = 'softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def lstm_model_n(hidden_size, nb_features, n_layers):
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    if n_layers <= 0 : n_layers = 1 # dont allow zero and non postive number of layers
    model = Sequential()
    # change input_shape[0] to None if want variable sequence length input tokens 
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, nb_features), name="lstm_1"))
    # adjust layer size of layer2 from hidden_size and observe effects on results
    for i in range(1, n_layers): model.add(LSTM(hidden_size, return_sequences=True, name=f"lstm_{str(i+1)}"))
    #Output 1 character with Dense layer of size 1 vector
    #model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def bi_lstm_model_n(hidden_size, nb_features, n_layers):
    if n_layers <= 0 : n_layers = 1 # dont allow zero and non postive number of layers
    model = Sequential()
    # change input_shape[0] to None if want variable sequence length input tokens
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True, input_shape=(None, nb_features))))

    # adjust layer size of layer2 from hidden_size and observe effects on results
    for i in range(n_layers-1): model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))

    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

