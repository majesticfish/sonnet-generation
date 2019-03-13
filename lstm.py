from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, \
        Dropout, TimeDistributed
from keras.callbacks import ModelCheckpoint
import preprocess
import numpy as np
class KerasBatchGenerator(object):
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_idx = 0
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start
                    self.current_idx = (self.current_idx + self.num_steps) % len(self.data)
                x[i, :] = self.data[self.current_idx:self.current_idx + \
                        self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + \
                        self.num_steps + 1]
                y[i, :, :] = to_categorical(temp_y,
                        num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y

def train_shakespeare(batch_size, n_epochs):
    # grab the data
    X_train, X_test, vocab, rev_dict = preprocess.get_data()

    # initialize starting conditions
    hidden_size = 150
    num_steps = 30

    model = Sequential()
    model.add(Embedding(vocab, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))

    # add dropout for regularization
    model.add(Dropout(0.5))

    model.add(TimeDistributed(Dense(vocab)))
    model.add(Activation('softmax'))

    # compilation
    model.compile(loss='categorical_crossentropy', optimizer='adam',
            metrics=['categorical_accuracy'])

    # for saving progress
    checkpointer = ModelCheckpoint(filepath='model-{epoch:02d}.hdf5',
            verbose=1)

    train_data_generator = KerasBatchGenerator(X_train, num_steps, batch_size, vocab)
    valid_data_generator = KerasBatchGenerator(X_test, num_steps, batch_size, vocab)

    model.fit_generator(train_data_generator.generate(),
            len(X_train)//(batch_size * num_steps), n_epochs,
            validation_data=valid_data_generator.generate(),
            validation_steps=len(X_test)//(batch_size * num_steps),
            callbacks=[checkpointer])

train_shakespeare(19, 50)
