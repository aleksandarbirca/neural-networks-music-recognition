import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, TimeDistributedDense
from keras.layers import LSTM, Embedding, GRU
import os
import glob
import numpy as np
from keras.optimizers import SGD, RMSprop
from sklearn.preprocessing import scale

config = {}
execfile("..\..\config.cfg", config)
GENRE_LIST = config["GENRE_LIST"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]

model = Sequential()


def compile_model():
    model.add(LSTM(output_dim=50, input_dim=1, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.15))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01, clipnorm=10)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print '\nNetwork created successfully and network model saved as file model.json.'


def train_network():
    x, y = read_mfcc(DATASET_DIR)
    x_test, y_test = read_mfcc(TEST_DIR)
    print '\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print '\nTraining network started\n'
    # x = normalize_cepstrum_coefficients(x)
    x = scale(x, axis=1, with_mean=True, with_std=True, copy=True)
    x_test = scale(x_test, axis=1, with_mean=True, with_std=True, copy=True)

    x_mean = x.mean()
    x_test_mean = x_test.mean()
    x -= x_mean
    x_test -= x_test_mean
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model.fit(x, y, nb_epoch=1000, batch_size=512, validation_data=(x_test, y_test))
    model.save_weights('..\..\data\weights.h5', overwrite=True)
    score = model.evaluate(x_test, y_test)
    print 'Network trained successfully and network weights saved as file weights.h5.'
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def normalize_cepstrum_coefficients(x):
    x[:, 0] = x[:, 0] / 20
    x[:, 1:13] = (x[:, 1:13] + 3) / 7
    return x


def read_mfcc(data_dir):
    x = []
    y = []
    for label, genre in enumerate(GENRE_LIST):
        for file in glob.glob(os.path.join(data_dir, genre, "*.ceps.npy")):
            print 'Extracting MFCC from ' + file
            ceps = np.load(file)
            num_ceps = len(ceps)
            x.append(np.mean(ceps[0:num_ceps], axis=0))
            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(x), np.array(y)


if __name__ == "__main__":
    compile_model()
    train_network()
