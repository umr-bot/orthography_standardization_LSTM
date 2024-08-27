from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed, Bidirectional, Embedding, Reshape
from tensorflow.keras import optimizers, metrics, backend as K
from keras.models import Model
import tensorflow as tf
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.333
session = tf.compat.v1.Session(config=config)

def softmax_regression(nb_features):
    model = Sequential()
    model.add(Input(shape=(None, nb_features), name='input_layer'))
    model.add(Dense(nb_features, activation = 'softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def softmax_regression_reg(nb_features, bias_reg,kernel_reg):
    """bias_reg and kernel reg are classes containing two objects L1,L2 each """
    with session.as_default():
        model = Sequential()
        model.add(Input(shape=(None, nb_features), name='input_layer'))
        model.add(Dense(nb_features, activation = 'softmax',bias_regularizer=(bias_reg),kernel_regularizer=(kernel_reg)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


