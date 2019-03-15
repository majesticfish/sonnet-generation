# Load LSTM network and generate text
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "data/shakespeare.txt"
f = open(filename)
lines = f.readlines()
raw_text = ""
for line in lines:
    if lines == '\n':
        continue
    try:
        int(line)
    except ValueError:
        raw_text = raw_text + line
raw_text = raw_text.lower()

# create mapping of unique chars to
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

for i in range(40*14):
    x = np.reshape(pattern,(1,len(pattern), 1))
    x = x / float(n_vocab)

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [int_to_char[value] for value in pattern]
    poem += int_to_char[pred_index]

    pattern.append(pred_index)
    pattern = pattern[1:len(pattern)]

print ("Poem:\n", poem)
