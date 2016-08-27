from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
import os
import glob
import numpy as np
import theano

config = {}
exec(open("..\..\config.cfg").read(), config)
GENRE_LIST = config["GENRE_LIST"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]


theano.config.optimizer = 'None'

model = Sequential()

rows = 42 # zavisi od semplovanja mfcc-a, za svali 100. je 42 (4133 mfcc-a u svakom fajlu posto je fajl 30 sekundi)
columns = 13

def compile_convolution2d_model():

    # 30 filtera, konvolucioni kernel 13*1 (rows 13 cols 1),  ulaz 42x13x1 (svaki 100. ceps)
    model.add(Convolution2D(30, 1, 13, border_mode='same', input_shape=(1, rows, columns)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(150))
    model.add(Activation('relu'))

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

    x = x.reshape(x.shape[0], 1, rows, columns)
    x_test = x_test.reshape(x_test.shape[0], 1, rows, columns)

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
            temp_signal = []
            ceps = np.load(file)
            temp_signal.extend(ceps[0::100])
            x.append(np.array(temp_signal))
            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(x), np.array(y)


if __name__ == "__main__":
    compile_convolution2d_model()
    train_network()
