# StackOverflow question: https://stackoverflow.com/questions/51123481/how-to-build-a-language-model-using-lstm-that-assigns-probability-of-occurence-f
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Reshape
#from tensorflow.keras.utils import Sequence
from keras.models import Sequential
import keras
import numpy as np
from utils import prepare_sentence, DataGenerator

def evaluate(sentence, model):
    # Compute probability of occurence of a sentence
    #sentence = data[0]
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

if __name__ == '__main__':
    # Data
#    data = ["Two little dicky birds",
#            "Sat on a wall,",
#            "One called Peter,",
#            "One called Paul.",
#            "Fly away, Peter,",
#            "Fly away, Paul!",
#            "Come back, Peter,",
#            "Come back, Paul."]
    with open("trigrams_v9/norm_foldset1/train") as f: train = [line.strip('\n') for line in f]
    with open("trigrams_v9/norm_foldset1/val") as f: val = [line.strip('\n') for line in f]

    # Preprocess data
    train_loader = DataGenerator(sentences=train, batch_size=4, shuffle=True)
    val_loader = DataGenerator(sentences=val, batch_size=4, shuffle=True)
    num_epochs = 100; hdd = 256
    train_steps = len(train_loader)//num_epochs
    val_steps = len(val_loader)//num_epochs

    # Define model
    model = Sequential()
    vocab_size = len(train_loader.vocab)
    #model.add(Reshape((vocab_size * (train_loader.maxlen-1), 1), input_shape=(train_loader.maxlen-1, vocab_size)))
    #model.add( Embedding(input_dim=vocab_size+1, output_dim=100, input_length=(train_loader.maxlen-1)*2) )
    #model.add(Reshape((train_loader.maxlen, 100*vocab_size)))
    model.add(Embedding(input_dim=vocab_size + 1,  # vocabulary size. Adding an
                        # extra element for <PAD> word
                        output_dim=5,  # size of embeddings
                        input_length=train_loader.maxlen - 1))  # length of the padded sequences
    model.add(LSTM(hdd))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy')

    # Train network
    model.fit(train_loader, steps_per_epoch=train_steps, validation_data=val_loader, validation_steps=val_steps, epochs=num_epochs, verbose=1)
    model.save("eng_trigram_foldset1.h")
    evaluate(data[0], model=model)
#    # Compute probability of occurence of a sentence
#    sentence = data[0]
#    tok = generator.tokenizer.texts_to_sequences([sentence])[0]
#    x_test, y_test = prepare_sentence(tok, generator.maxlen)
#    x_test = np.array(x_test)
#    y_test = np.array(y_test) - 1  # The word <PAD> does not have a class
#    p_pred = model.predict(x_test)
#    vocab_inv = {v: k for k, v in generator.vocab.items()}
#    log_p_sentence = 0
#    for i, prob in enumerate(p_pred):
#        word = vocab_inv[y_test[i] + 1]  # Index 0 from vocab is reserved to <PAD>
#        history = ' '.join([vocab_inv[w] for w in x_test[i, :] if w != 0])
#        prob_word = prob[y_test[i]]
#        log_p_sentence += np.log(prob_word)
#        print('P(w={}|h={})={}'.format(word, history, prob_word))
#    print('Prob. sentence: {}'.format(np.exp(log_p_sentence)))
