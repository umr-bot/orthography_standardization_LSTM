from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from feature_analyzer import get_iter
import os

hidden_size=128
train_batch_size, val_batch_size = 32,32

train_iter, val_iter, train_tokens, val_tokens, nb_total_chars = get_iter(epoch=0)

model = Sequential()
model.add(Input(shape=(None, nb_total_chars), name='input_layer'))
#model.add(Dense(hidden_size, activation = 'sigmoid',input_dim = hidden_size))
#model.add(Dense(hidden_size, activation = 'sigmoid',input_dim = hidden_size))
model.add(Dense(nb_total_chars, activation = 'softmax',input_dim = hidden_size))

#adam = tensorflow.keras.optimizers.Adam(lr=0.001, decay=0.0)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

train_steps = len(train_tokens) // train_batch_size
val_steps = len(val_tokens) // val_batch_size
print("Number of train_steps:",train_steps)
print("Number of val_steps:",val_steps)
start=0
for epoch in range(start,20):
    print(f"Epoch {str(epoch+1)}")
    train_loader, val_loader, train_tokens, val_tokens, nb_total_chars = get_iter(epoch=epoch)
    history = model.fit(train_loader,
                        steps_per_epoch=train_steps,
                        epochs=1, verbose=1,
                        validation_data=val_loader,
                        validation_steps=val_steps)    

    # Save the model at end of each epoch.
    model_file = '_'.join(['seq2seq', 'epoch', str(epoch + 1)]) + '.h5'
    save_dir = 'eng_checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if (epoch+1) % 10 == 0:
        save_path = os.path.join(save_dir, model_file)
        print('Saving full model to {:s}'.format(save_path))
        model.save(save_path)
    fn = save_dir+"/history.txt"
    #for history in score:
    with open(fn,'a') as f:
        f.write(str(history.history['loss'][0]) + ',')
        f.write(str(history.history['val_loss'][0]) + ',')
        f.write(str(history.history['accuracy'][0]) + ',')
        f.write(str(history.history['val_accuracy'][0]) + ',')
        #f.write(str(history.history['precision'][0]) + ',')
        #f.write(str(history.history['recall'][0]) + ',')
        #f.write(str(history.history['f1_score'][0])+',')
        #f.write(str(history.history['val_precision'][0]) + ',')
        #f.write(str(history.history['val_recall'][0]) + ',')
        #f.write(str(history.history['val_f1_score'][0]))
       
        f.write('\n')

