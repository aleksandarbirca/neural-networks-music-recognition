from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Convolution1D, MaxPooling1D, Flatten
import os
import glob
import numpy as np
from sklearn.preprocessing import scale
import theano
#theano.config.optimizer = 'None'

config = {}
exec(open("..\..\config.cfg").read(), config)
GENRE_LIST = config["GENRE_LIST"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]

model = Sequential()


def compile_convolution1d_model():
    model.add(Embedding(120, 120, input_length=13))
    model.add(Convolution1D(120, 3, border_mode='same'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model.json.')


def train_network():
    x, y = read_mfcc(DATASET_DIR)
    x_test, y_test = read_mfcc(TEST_DIR)
    print ('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print ('\nTraining network started\n')

    x = scale(x, axis=1, with_mean=True, with_std=True, copy=True)
    x_test = scale(x_test, axis=1, with_mean=True, with_std=True, copy=True)

    model.fit(x, y, nb_epoch=1000, batch_size=128, validation_data=(x_test, y_test))
    model.save_weights('..\..\data\weights.h5', overwrite=True)
    score = model.evaluate(x_test, y_test)
    print ('\nNetwork trained successfully and network weights saved as file weights.h5.')
    print('\nTest score:', score[0])
    print('\nTest accuracy:', score[1])


def read_mfcc(data_dir):
    x = []
    y = []
    for label, genre in enumerate(GENRE_LIST):
        for file in glob.glob(os.path.join(data_dir, genre, "*.ceps.npy")):
            print ('Extracting MFCC from ' + file)
            ceps = np.load(file)
            num_ceps = len(ceps)
            temp_signal = []
            temp_signal.extend(np.mean(ceps[0:num_ceps], axis=0))
            #temp_signal.extend(np.min(ceps[0:num_ceps], axis=0))
            #temp_signal.extend(np.max(ceps[0:num_ceps], axis=0))
            x.append(temp_signal)
            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(x), np.array(y)


if __name__ == "__main__":
    compile_convolution1d_model()
    train_network()
