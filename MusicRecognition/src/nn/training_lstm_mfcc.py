from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
import os
import numpy as np
import glob
from sklearn.preprocessing import scale


config = {}
exec(open("..\..\config.cfg").read(), config)
GENRE_LIST = config["GENRE_LIST"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]

model = Sequential()

num_of_mfcc = 100


def compile_lstm_model():
    model.add(LSTM(512, return_sequences=True, activation='tanh', input_shape=(num_of_mfcc, 13)))
    model.add(Dropout(0.1))
    model.add(LSTM(512, activation='tanh', return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    #optimizer = RMSprop(lr=0.9)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model.json.')


def train_network():
    x, y = read_mfcc(DATASET_DIR)
    x_test, y_test = read_mfcc(TEST_DIR)

    x = np.reshape(x, (x.shape[0], x.shape[1], 13))
    x_test = np.reshape(x_test, (x_test.shape[0], x.shape[1], 13))

    print ('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print ('\nTraining network started\n')
    model.fit(x, y, nb_epoch=1000, batch_size=16, validation_data=(x_test, y_test))
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
            ceps = ceps[0:num_of_mfcc]
            normalize_cepstrum_coefficients(ceps)
            x.append(np.array(ceps))
            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(x), np.array(y)


def normalize_cepstrum_coefficients(x):
    x[:, 0] = x[:, 0] / 20
    x[:, 1:13] = x[:, 1:13] / 3
    return x

if __name__ == "__main__":
    compile_lstm_model()
    train_network()
