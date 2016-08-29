from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
import os
import scipy.io.wavfile as wavfile
import numpy as np
import glob


config = {}
exec(open("..\..\config.cfg").read(), config)
GENRE_LIST = config["GENRE_LIST"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]

model = Sequential()

length = 1000
timesteps = 3
inputdim = 1


def compile_recurrent_model():
    model.add(LSTM(output_dim=50, input_dim=inputdim, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(120, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01, clipnorm=10)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model.json.')


def train_network():
    x, y = read_wav(DATASET_DIR)
    x_test, y_test = read_wav(TEST_DIR)

    print (x.shape)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x.shape[1], 1))

    print ('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print ('\nTraining network started\n')
    model.fit(x, y, nb_epoch=1000, batch_size=128, validation_data=(x_test, y_test))
    model.save_weights('..\..\data\weights.h5', overwrite=True)
    score = model.evaluate(x_test, y_test)
    print ('\nNetwork trained successfully and network weights saved as file weights.h5.')
    print('\nTest score:', score[0])
    print('\nTest accuracy:', score[1])


def read_wav(data_dir):
    x = []
    y = []
    for label, genre in enumerate(GENRE_LIST):
        for path in glob.glob(os.path.join(data_dir, genre, "*.wav")):
            print 'Reading' + path + '.'
            data = wavfile.read(path)
            normalized_signal = data[1].astype('float32') / 32767.0  # range [-1, 1]
            normalized_signal = normalized_signal[0::1000]
            normalized_signal = normalized_signal[0:500]
            x.append(normalized_signal)

            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(x), np.array(y)

if __name__ == "__main__":
    compile_recurrent_model()
    #read_wav(TEST_DIR)
    train_network()
