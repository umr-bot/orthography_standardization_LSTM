from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed, Bidirectional, Embedding, Reshape
from tensorflow.keras import optimizers, metrics, backend as K

from keras.models import Model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Activation
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import SGD, Nadam, RMSprop, Adam
from keras.layers import TimeDistributed as TD

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

def softmax_regression(hidden_size, nb_features):
    model = Sequential()
    model.add(Input(shape=(None, nb_features), name='input_layer'))
    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall,f1_score], run_eagerly=True)
    return model

def softmax_regression_2(hidden_size, nb_features):
    model = Sequential()
    model.add(Input(shape=(None, nb_features), name='input_layer'))
    model.add(Dense(hidden_size, activation = 'sigmoid',input_dim = hidden_size))
    #model.add(Dense(hidden_size, activation = 'sigmoid',input_dim = hidden_size))
    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    #adam = tensorflow.keras.optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall,f1_score], run_eagerly=True)
    return model

def lstm_model(hidden_size, nb_features):
    model = Sequential()
    # change input_shape[0] to None if want variable sequence length input tokens 
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, nb_features)))

    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall,f1_score], run_eagerly=True)

    return model

def lstm_model_n(hidden_size, nb_features, n_layers):
    if n_layers <= 0 : n_layers = 1 # dont allow zero and non postive number of layers
    model = Sequential()
    # change input_shape[0] to None if want variable sequence length input tokens 
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, nb_features)))
    # adjust layer size of layer2 from hidden_size and observe effects on results
    for i in range(n_layers-1): model.add(LSTM(hidden_size, return_sequences=True))
    #Output 1 character with Dense layer of size 1 vector
    #model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall,f1_score], run_eagerly=True)

    return model

def lstm_model_n_dropout(hidden_size, nb_features, n_layers, dropout):
    if n_layers <= 0 : n_layers = 1 # dont allow zero and non postive number of layers
    model = Sequential()
    # change input_shape[0] to None if want variable sequence length input tokens 
    model.add(LSTM(hidden_size, dropout=dropout, return_sequences=True, input_shape=(None, nb_features)))
    # adjust layer size of layer2 from hidden_size and observe effects on results
    for i in range(n_layers-1): model.add(LSTM(hidden_size, dropout=dropout, return_sequences=True))
    #Output 1 character with Dense layer of size 1 vector
    #model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall,f1_score], run_eagerly=True)

    return model

def bi_lstm_model_n(hidden_size, nb_features, n_layers):
    if n_layers <= 0 : n_layers = 1 # dont allow zero and non postive number of layers
    model = Sequential()
    # change input_shape[0] to None if want variable sequence length input tokens 
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True, input_shape=(None, nb_features))))

    # adjust layer size of layer2 from hidden_size and observe effects on results
    for i in range(n_layers-1): model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
    #Output 1 character with Dense layer of size 1 vector
    #model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall,f1_score], run_eagerly=True)

    return model

def lstm_embed(hidden_size, nb_features, n_layers, embed_output_dim, maxlen):
    if n_layers <= 0 : n_layers = 1 # dont allow zero and non postive number of layers
    model = Sequential()
    # flatten input tuple (max_word_len, nb_features), where nb_features = num of char types
    model.add(Reshape((nb_features * maxlen, 1), input_shape=(maxlen, nb_features)))
    model.add( Embedding(input_dim=nb_features+1, output_dim=embed_output_dim, input_length=maxlen*2) )
    model.add(Reshape((maxlen, embed_output_dim*nb_features)))
    # change input_shape[0] to None if want variable sequence length input tokens 
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(maxlen, embed_output_dim*nb_features)))
    #model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, nb_features)))
    # adjust layer size of layer2 from hidden_size and observe effects on results
    for i in range(n_layers-1): model.add(LSTM(hidden_size, return_sequences=True))
    #Output 1 character with Dense layer of size 1 vector
    #model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    
    model.add(Dense(nb_features, activation = 'softmax',input_dim = hidden_size))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',precision,recall,f1_score], run_eagerly=True)

    return model

#def lstm_2_embed(nb_features, embed_output_dim, maxlen, opt='adam'):
#    print("Building Model")
#    inp_text = Input(shape=(maxlen,))
#    x0 = Embedding(input_dim=nb_features, output_dim=embed_output_dim, mask_zero=False)(inp_text)
#    
#    x = LSTM(32, return_sequences=True)(x0)
#    x1 = Activation("relu")(x)
#    x = LSTM(64, return_sequences=True)(x1)
#    x2 = Activation("relu")(x)
#    
#    x = concatenate([x0, x1, x2])#"Skip connections"
#    y = TD(Dense(nb_features, activation="softmax"))(x)
#    
#    model = Model(inputs=[inp_text],outputs=y)
#
#    if opt=="sgd" :optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
#    if opt=="rms" :optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#    if opt=="adam" :optimizer= Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#    if opt=="nadam" :optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#    
#    #note: there seems to be updates to how keras deals with sample weighting
#    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
#              sample_weight_mode="temporal",weighted_metrics=['accuracy'])
#    model.summary()
#    return model
