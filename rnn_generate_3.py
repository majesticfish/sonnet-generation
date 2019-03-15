# Load LSTM network and generate text
import sys
import numpy as np
from random import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def usage():
    print("python rnn_generate.py weights_file temp")

if len(sys.argv) != 3:
    usage()
    sys.exit()
# load ascii text and covert to lowercase
filename = "data/shakespeare + rap.txt"
f = open(filename)
lines = f.readlines()
raw_text = ""
for line in lines:
    if line == '\n':
        continue
    try:
        int(line)
    except ValueError:
        raw_text = raw_text + line
raw_text = raw_text.lower()

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 40
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(500, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(500, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(500))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# load the network weights
filename = sys.argv[1]
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# seed
text_seed = "shall i compare thee to a summer's day?\n"
pattern = [char_to_int[char] for char in text_seed]
print("PATTERN:", pattern)
testing = "".join([int_to_char[value] for value in pattern])
print("THE REVERSE", testing)

poem = ""

temp = float(sys.argv[2])

def sample(prediction_dist, temp):
    '''Applies temperature to a softmax output. Returns a sample from the
    prediction distribution with temperature

    Parameters:
        prediction_dist: a 1 X N array of predictions
        temp: a float describing the temperature
    '''

    prediction = np.log(prediction_dist) / temp
    prediction = np.exp(prediction[0])
    prediction = prediction/np.sum(prediction)
    r = random()
    pred = 0
    while r > 0:
        r = r - prediction[pred]
        pred = pred + 1
    return pred - 1

for i in range(40*14):
    x = np.reshape(pattern,(1,len(pattern), 1))
    x = x / float(n_vocab)

    prediction_dist = model.predict(x, verbose=0)
    pred_index = sample(prediction_dist, temp)

    poem += int_to_char[pred_index]

    pattern.append(pred_index)
    pattern = pattern[1:len(pattern)]



'''
# generate characters
for i in range(1000):
 #   x = np.reshape(npattern, len(npattern), 1)
  #  print("THIS IS X", x)
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    poem += result
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]'''
print ("Poem:\n", poem)
