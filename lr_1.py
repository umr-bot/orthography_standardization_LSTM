import torch
import torch.nn as nn
import torch.optim as optim
#from torchtext.legacy import data
#from torchtext.legacy import datasets
from feature_analyzer import get_iter
# Define the field for the text data
#TEXT = data.Field(tokenize='spacy')

# Load the IMDB dataset and split it into training and test sets
#train_data, test_data = datasets.IMDB.splits(TEXT)

# Build the vocabulary from the training data
#TEXT.build_vocab(train_data)

# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# Set the device to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, test_iter, train_tokens, val_tokens = get_iter(epoch=0)

# Set the hyperparameters
#input_dim = len(TEXT.vocab)
temp_batch = next(train_iter)
input_dim = len(temp_batch[0][0][0][0]) #len(set(tuple(train_tokens)).union(set(tuple(val_tokens))))
output_dim = 2 #len(temp_batch[0][0])*len(temp_batch[0][0][0]) #2
lr = 0.001
batch_size = 64
epochs = 10

# Define the iterators for the training and test sets
#train_iter, test_iter = data.BucketIterator.splits(
#    (train_data, test_data), batch_size=batch_size, device=device)

# Initialize the model and optimizer
model = LogisticRegression(input_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(epochs):
 
    train_iter, test_iter, blank, blank = get_iter(epoch=epoch)    
    for batch in train_iter:
        optimizer.zero_grad()
        #x = batch.text.to(device)
        #y = batch.label.to(device)
        x = torch.from_numpy(batch[0][0]).to(device) # seq2seq encoder data
        y = torch.from_numpy(batch[1]).to(device) # seq2seq target data
        y_pred = model(x)
        loss = nn.CrossEntropyLoss()(y_pred, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate the model on the test set
#    with torch.no_grad():
#        correct = 0
#        total = 0
#        for batch in test_iter:
#            x = batch.text.to(device)
#            y = batch.label.to(device)
#            y_pred = model(x)
#            _, predicted = torch.max(y_pred.data, 1)
#            total += y.size(0)
#            correct += (predicted == y).sum().item()
#        accuracy = 100 * correct / total
#        print('Epoch: {}, Test Accuracy: {}%'.format(epoch+1, accuracy))

