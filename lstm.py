from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, \
        Dropout, TimeDistributed
from keras.callbacks import ModelCheckpoint
import preprocess
import numpy as np
import tensorflow as tf
import os
tf.enable_eager_execution()


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

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def train_shakespeare(batch_size, n_epochs):
    # grab the data
    text_as_int, text, vocab, char2idx, idx2char = preprocess.get_data()

    # The maximum length sentence we want for a single input in characters
    seq_length = 40
    examples_per_epoch = len(text)//seq_length

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    seq = char_dataset.batch(seq_length+1, drop_remainder=True)

    for i in range(1, seq_length, 3):
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int[i:])
        sequence = char_dataset.batch(seq_length+1, drop_remainder=True)
        seq = seq.concatenate(sequence)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = seq.map(split_input_target)

    for input_example, target_example in  dataset.take(1):
        print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
        print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
    steps_per_epoch = examples_per_epoch//batch_size

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    # The embedding dimension
    embedding_dim = 156

    # Number of RNN units
    rnn_units = 1024

    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNGRU
    else:
        import functools
        rnn = functools.partial(
            tf.keras.layers.GRU, recurrent_activation='sigmoid')
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
      model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(embedding_dim, return_sequences=True),
        tf.keras.layers.Dense(vocab_size),
      ])
      return model
    model = build_model(
        vocab_size = len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=batch_size)

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
        print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
        example_batch_loss  = loss(target_example_batch, example_batch_predictions)
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("scalar_loss:      ", example_batch_loss.numpy().mean())
    model.compile(optimizer = tf.train.AdamOptimizer(), loss = loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    history = model.fit(dataset.repeat(), epochs=n_epochs, steps_per_epoch=5*steps_per_epoch,
            callbacks=[checkpoint_callback])

def grab_model(filename):
    # grab the data
    text_as_int, text, vocab, char2idx, idx2char = preprocess.get_data()

    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.LSTM(embedding_dim, return_sequences=True),
            tf.keras.layers.Dense(vocab_size),
        ])
        return model
    # The embedding dimension
    embedding_dim = 156

    model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim,
            rnn_units=0, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
    model.build(tf.TensorShape([1,None]))
    model.summary()

    def generate_text(model, start_string, temp=None):
        # number of characters to generate
        num_generate = 1000

        # converting our start string to numbers
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # string to store generation
        text_generated = []

        # Low temp results in more predictable text
        # high temp results in more su rprising text
        # experiment
        if temp == None:
            temp = 1.0

        model.reset_states()

        for i in range(num_generate):
            predictions = model(input_eval)
            # remove batch dimension
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temp
            predicted_id = \
                    tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

            # we pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])
        return (start_string + ''.join(text_generated))

    return generate_text(model, "shall i compare thee to a summer's day?\n")
