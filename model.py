import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras import optimizers, metrics, backend as K
# For use with truncated metrics,
# take maxlen from the validation set.
# Hacky and hard-coded for now.
VAL_MAXLEN = 16

def truncated_acc(y_true, y_pred):
#    print("TA y_true len",len(y_true))
#    print("TA y_true\n",y_true)
    y_true = y_true[:, :VAL_MAXLEN, :]
    y_pred = y_pred[:, :VAL_MAXLEN, :]
    
    acc = metrics.categorical_accuracy(y_true, y_pred)
    return K.mean(acc, axis=-1)


def truncated_loss(y_true, y_pred):
#    print("TL y_true len",len(y_true))
#    print("TL y_true\n",y_true)
    y_true = y_true[:, :VAL_MAXLEN, :]
    y_pred = y_pred[:, :VAL_MAXLEN, :]
    
    loss = K.categorical_crossentropy(
        target=y_true, output=y_pred, from_logits=False)
    return K.mean(loss, axis=-1)

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    # K.clip is used in the case of non-binary classification
    # where elements of y_true and y_pred some floating point number
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

def seq2seq(hidden_size, nb_input_chars, nb_target_chars):
    """Adapted from:
    https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
    hidden_size: Number of memory cells in encoder and decoder models
    nb_input_chars: number of input sequences, in train_val.py 's case, chars
    nb_target_chars number of output sequences, in train_val.py, it is chars
    """
    
    # Define the main model consisting of encoder and decoder.
    encoder_inputs = Input(shape=(None, nb_input_chars), name='encoder_data')
    model.add(Dense(nb_target_chars, activation = 'sigmoid',input_dim = hidden_size))
    # here encoder outputs contains three things:{(all h states),(last h state),(last c state)}
    #encoder_outputs_1 = encoder_lr_1(encoder_inputs)

    #adam = tensorflow.keras.optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy', recall, f1_score], run_eagerly=True)
    
    return model
