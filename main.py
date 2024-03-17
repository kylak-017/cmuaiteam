import torch
import torch.functional as F
import torch.nn as nn
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

import re
import numpy as np
import time
import pickle
import pandas as pd
from helpers import pickle_helpers as ph
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Visualizing The Data


#Using PANDAS, importing the pickle data file

data = pd.read_pickle("data/EmoryNLP.pkl")
print(data)

with open('data/EmoryNLP.pkl', 'rb') as f:
    data = pickle.load(f)

collectList = {}
#{sentence, annotation}
#ex) {"hello", 3}

emotionDictionary = data[0]
vocabDictionary = data[2]

for i in range(len(emotionDictionary)):
    keysE = list(emotionDictionary.keys()) #has a list of keys from the diciontary, for example: 4_4_1, 4_4_2...
    keysV = list(vocabDictionary.keys()) 
    emotionList = emotionDictionary[keysE[i]] # has a list of values inside each key, the one-hot encoding vecotrs
    vocabList = vocabDictionary[keysE[i]]
    for j in range(len(emotionList)): #for each emotion of each sentence
        collectList[vocabList[j]] = emotionList[j]


'''
      for k in range (0,10): #getting value
            if(emotionList[j][k] == 1):
            '''


print(collectList)







    


#data2 = ph.load_from_pickle(directory = "")
#plotting specific array from pickle file (pickle file has a collection of all the tensors saved in arrays)
#data.emotions.value_counts().plot.bar()
#returns the first 10 rows of the dataframe, which is the tabular data storing the actual data
data.head(10)

#data_merged = data + data2

#Preprocessing Data

#Tokenization: Before creating dictionaries, formatting
#Tokens: pretty much words, individual smaller units of the natural language data set
# retain only text that contain less that 70 tokens to avoid too much padding
# The data dictionary uses token_size array as a key, applies a function lambda(miscellanous fucntions with unlim parameters and 1 argument)
data["token_size"] = data["text"].apply(lambda x: len(x.split(' ')))
data = data.loc[data['token_size'] < 70].copy()

# Sampling: selecting specific subsets of datasets
#n=50000 --> the 50000th row of the dataset
data = data.sample(n=50000)

#Constructing Vocabulary and Index-Word Mapping
class ConstructVocab():
    #constructor, don't need seperate instance variable for python
    def __init__(self, sentences):
        self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
    

    def create_index(self): #self parameter utilizes variables in the class, all of the instance variables
        for s in self.sentences:
            # update with individual tokens
            self.vocab.update(s.split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0 (mapped to index 0)
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        #for machine to understand
        #enumerate: counting the iterations by assigning indexes to the words
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        #for humans to read after words are interpretetd by machine
        #from the sections in the word2idx items list (the possible responses)
        for word, index in self.word2idx.items():
            self.idx2word[index] = word  

#Forward Propogation
            
#Setting up the input data from the tensors imported and tokenized
#construct vocab and indexing
#changes the dataframe to lists to feed in as inout (2D lists)
inputs = ConstructVocab(data["text"].values.tolist())

# examples of what is in the vocab
inputs.vocab[0:10]

# vectorize to tensor
#converting the "text" row of data(dataFrame) to list of values
input_tensor = [[inputs.word2idx[s] for s in es.split(' ')]  for es in data["text"].values.tolist()]
# examples of what is in the input tensors
input_tensor[0:2]

#Padding is usually used so that the shape or form of the data/tensor is in the same way. (convolutional layers)

#function rhat returns max value for all the tensors saved in length of input_tensor
def max_length(tensor):
    return max(len(t) for t in tensor)
# calculate the max_length of input tensor
max_length_inp = max_length(input_tensor)
print(max_length_inp)
#adding zeres to pad the data lengths
def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

# inplace padding
input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor]
input_tensor[0:2]

#Binarization

### convert targets to one-hot encoding vectors(there is only one unique number in a whole row of '0's)
#used to locate which category the specific character is in. ex) if categorical # is 1, then index 1 should have the unique key
emotions = list(set(data.emotions.unique()))
num_emotions = len(emotions)
# binarizer
mlb = preprocessing.MultiLabelBinarizer() #transforms to binary seqeunce (categorical #s)
#intersection between column emotions and data in the dataFrame, and finds the emotions in the dataFrame that intersect with the predefined emotions (by matching categorical # indexes)
data_labels =  [set(emos) & set(emotions) for emos in data[['emotions']].values]
print(data_labels)

bin_emotions = mlb.fit_transform(data_labels) #identifies each item(of set arrays of emotions) using a binary matrix
target_tensor = np.array(bin_emotions.tolist()) # matrix of possible labels
target_tensor[0:2] 
data[0:2]

get_emotion = lambda t: np.argmax(t) #returns argmax of t, and argmax finds the index of the maximum value in an array
get_emotion(target_tensor[0])

emotion_dict = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}
emotion_dict[get_emotion(target_tensor[0])]

#Splitting Data

#Both types of data will be pre-processed
#input: user's text (predicted emotion) (predicted value)
#target: emotion (actual emotion) (true value)

#train: 80, test: 20 (train: training data, test: unseen data)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
#this 20% for test, which is basically validation test data 

#50,50 out of the unseen data (50% for the training, 50% for the unseen data)
input_tensor_val, input_tensor_test, target_tensor_val, target_tensor_test = train_test_split(input_tensor_val, target_tensor_val, test_size=0.5)
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val), len(input_tensor_test), len(target_tensor_test)

#Loading Data
TRAIN_BUFFER_SIZE = len(input_tensor_train)
VAL_BUFFER_SIZE = len(input_tensor_val)
TEST_BUFFER_SIZE = len(input_tensor_test)
BATCH_SIZE = 64
TRAIN_N_BATCH = TRAIN_BUFFER_SIZE // BATCH_SIZE
VAL_N_BATCH = VAL_BUFFER_SIZE // BATCH_SIZE
TEST_N_BATCH = TEST_BUFFER_SIZE // BATCH_SIZE

embedding_dim = 256
units = 1024
vocab_inp_size = len(inputs.word2idx)
target_size = num_emotions

# convert the data to tensors and pass to the Dataloader 
# to create an batch iterator

class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len
    
    def __len__(self):
        return len(self.data)
    
train_dataset = MyData(input_tensor_train, target_tensor_train)
val_dataset = MyData(input_tensor_val, target_tensor_val)
test_dataset = MyData(input_tensor_test, target_tensor_test)

train_dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)
val_dataset = DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)
val_dataset.batch_size


#Forward propogation

#comes up with a predicted value using weights that were calculated from the layers that took the input functions
class emotion(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_layers, batch_sz, output_size):
        self.vocab_size = vocab_size #length of input that was converted to index (wordi2x)
        self.embedding_dim = embedding_dim #change the embeddings dimenisons based on the pkl file
        self.hidden_layers = hidden_layers
        self.batch_sz = batch_sz

        #layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim) #the actual tables, the dimensions of the neural network
        self.dropout = nn.Dropout(p=0.5) #dropout probability can change, dropout: if there are too many nodes, you make them "dropout" --> faster training time, can adjust the weights and the inputs of the activation function (p<=0.5)

        #i can change:
        #activation function types
        #loss functions
        #optimization function =  gradient descent
        #dropout
        #dimensions
        #pkl file
        #annotations
        #preprocessing
        

### sort batch function to be able to use with pad_packed_sequence
def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = emotion(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
model.to(device)

# obtain one sample from the data iterator
it = iter(train_dataset)
x, y, x_len = next(it)

# sort the batch first to be able to use with pac_pack sequence
xs, ys, lens = sort_batch(x, y, x_len)

print("Input size: ", xs.size())

output, _ = model(xs.to(device), lens, device)
print(output.size())



### Enabling cuda
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda else "cpu")
model = emotion(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
model.to(device)

### loss criterion and optimizer for training
criterion = nn.CrossEntropyLoss() # the same as log_softmax + NLLLoss
optimizer = torch.optim.Adam(model.parameters())


#changeable -->  Mean Absolute Error/Quadratic, RELU/Sigmoid
def loss_function(y, prediction):
    """ CrossEntropyLoss expects outputs and class indices as target """
    # convert from one-hot encoding to class indices
    target = torch.max(y, 1)[1]
    loss = criterion(prediction, target) 
    return loss   #TODO: refer the parameter of these functions as the same
    

#changeable --> Mean Absolute Error/
def accuracy(target, logit):
    ''' Obtain accuracy for training round '''
    target = torch.max(target, 1)[1] # convert from one-hot encoding to class indices
    corrects = (torch.max(logit, 1)[1].data == target).sum()
    accuracy = 100.0 * corrects / len(logit)
    return accuracy

#changeable
EPOCHS = 30 #10

for epoch in range(EPOCHS):
    start = time.time()
    
    ### Initialize hidden state
    # TODO: do initialization here.
    total_loss = 0
    train_accuracy, val_accuracy = 0, 0
    
    ### Training
    for (batch, (inp, targ, lens)) in enumerate(train_dataset):
        loss = 0
        predictions, _ = model(inp.permute(1 ,0).to(device), lens, device) # TODO:don't need _   
              
        loss += loss_function(targ.to(device), predictions)
        batch_loss = (loss / int(targ.shape[1]))        
        total_loss += batch_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_accuracy = accuracy(targ.to(device), predictions)
        train_accuracy += batch_accuracy
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Val. Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.cpu().detach().numpy()))
            
    ### Validating
    for (batch, (inp, targ, lens)) in enumerate(val_dataset):        
        predictions,_ = model(inp.permute(1, 0).to(device), lens, device)        
        batch_accuracy = accuracy(targ.to(device), predictions)
        val_accuracy += batch_accuracy
    
    print('Epoch {} Loss {:.4f} -- Train Acc. {:.4f} -- Val Acc. {:.4f}'.format(epoch + 1, 
                                                             total_loss / TRAIN_N_BATCH, 
                                                             train_accuracy / TRAIN_N_BATCH,
                                                             val_accuracy / VAL_N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))






#Backward Propogation