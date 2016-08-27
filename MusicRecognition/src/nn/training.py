from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Convolution1D, MaxPooling1D, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
import os
import glob
import numpy as np
from sklearn.preprocessing import scale
import theano

config = {}
exec(open("..\..\config.cfg").read(), config)
GENRE_LIST = config["GENRE_LIST"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]

#theano.config.blas.ldflags = "-LC:\mreze\openblas -lopenblas"
#theano.config.device = 'cpu'
#theano.config.floatX = 'float32'

theano.config.optimizer = 'None'

model = Sequential()


def compile_basic_model():
    sgd = SGD(lr=0.9, decay=1e-6, momentum=0.8, nesterov=True)
    model.add(Dense(120, input_dim=39))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model.json.')


def compile_recurrent_model():
    model.add(LSTM(output_dim=50, input_dim=1, return_sequences=True))
    #model.add(Dropout(0.1))
    model.add(LSTM(120, return_sequences=False))
    #model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01, clipnorm=10)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model.json.')


def compile_convolution1d_model():
    model.add(Embedding(1, 800, input_length=13))
    model.add(Convolution1D(5, 3, border_mode='same'))
    model.add(MaxPooling1D(pool_length=2))
    #model.add(Convolution1D(32, 8, border_mode='same'))
    #model.add(MaxPooling1D(pool_length=4))
    model.add(Flatten())
    #model.add(Dense(100))
    #model.add(Activation('tanh'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model.json.')


rows = 42
columns = 13

def compile_convolution2d_model():

    # prvi conv sloj se sastoji od 30 filtera 13x1, ulaz su slike 42x13x1
    # aktivaciona funkcija je ReLU
    model.add(Convolution2D(30, 1, 13, border_mode='same', input_shape=(1, rows, columns))) # 30 filtera, konvolucioni kernel 13*1 (rows 13 cols 1)
    model.add(Activation('relu'))

    model.add(Convolution2D(30, 1, 13, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))

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

    #x = scale(x, axis=1, with_mean=True, with_std=True, copy=True)
    #x_test = scale(x_test, axis=1, with_mean=True, with_std=True, copy=True)

    #x_mean = x.mean()
    #x_test_mean = x_test.mean()
    #x -= x_mean
    #x_test -= x_test_mean

    #x = normalize_cepstrum_coefficients(x)
    #x_test = normalize_cepstrum_coefficients(x_test)

    #x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

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
            #temp_signal = [ceps[0::100]]
            temp_signal = []
            temp_signal.extend(ceps[0::100])
            #temp_signal.extend(np.mean(ceps[0:num_ceps], axis=0))
            #temp_signal.extend(np.min(ceps[0:num_ceps], axis=0))
            #temp_signal.extend(np.max(ceps[0:num_ceps], axis=0))
            x.append(temp_signal)
            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(x), np.array(y)


def normalize_cepstrum_coefficients(x):
    x[:, 0] = x[:, 0] / 20
    x[:, 1:13] = (x[:, 1:13] + 3) / 7
    return x

if __name__ == "__main__":
    #compile_basic_model()
    #compile_recurrent_model()
    #compile_convolution1d_model()
    compile_convolution2d_model()
    train_network()
