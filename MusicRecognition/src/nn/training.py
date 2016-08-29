import numpy
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, TimeDistributedDense
from keras.layers import LSTM, Embedding, GRU, Convolution1D, MaxPooling1D
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
    model.add(GRU(output_dim=50, input_dim=1, return_sequences=True))
    # model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(GRU(100, return_sequences=True))
    model.add(Dropout(0.20))
    model.add(GRU(50, return_sequences=False))
    # model.add(Activation('tanh'))
    model.add(Dropout(0.15))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print '\nNetwork created successfully and network model saved as file model.json.'


def train_network():
    x, y = read_mfcc(DATASET_DIR)
    x_test, y_test = read_mfcc(TEST_DIR)

    print '\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print '\nTraining network started\n'

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x.shape[1], 1))

    model.fit(x, y, nb_epoch=1000, batch_size=128, validation_data=(x_test, y_test))
    model.save_weights('..\..\data\weights.h5', overwrite=True)
    score = model.evaluate(x_test, y_test)
    print 'Network trained successfully and network weights saved as file weights.h5.'
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def read_mfcc(data_dir):
    x = []
    y = []
    for label, genre in enumerate(GENRE_LIST):
        for file in glob.glob(os.path.join(data_dir, genre, "*.ceps.npy")):
            print 'Extracting MFCC from ' + file
            ceps = np.load(file)
            num_ceps = len(ceps)
            temp_signal = []
            # temp_signal = ceps.ravel()
            temp_signal.extend(np.mean(ceps[0:num_ceps], axis=0))
            temp_signal.extend(np.min(ceps[0:num_ceps], axis=0))
            temp_signal.extend(np.max(ceps[0:num_ceps], axis=0))
            # temp_signal.extend(np.std(ceps[0:num_ceps], axis=0))
            temp_signal.extend(np.var(ceps[0:num_ceps], axis=0))
            x.append(temp_signal)
            g = np.zeros(3)
            g[label.real] = 1
            y.append(g)
    x = np.array(x)
    x = scale(x, axis=1, with_mean=True, with_std=True, copy=True)
    x = (x - x.min()) / (x.max() - x.min())
    x_mean = x.mean()
    x -= x_mean
    return x, np.array(y)

if __name__ == "__main__":
    compile_model()
    train_network()
