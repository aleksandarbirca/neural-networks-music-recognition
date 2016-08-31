from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import os
import glob
import numpy as np


##########################################################################################################
# This implementation is not expected to give a good results and it is here only for comparison purpose. #
##########################################################################################################

config = {}
exec(open("..\..\config.cfg").read(), config)
GENRES = config["GENRES"]
GENRES_ALL = config["GENRES_ALL"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]


# Number of coefficients.
# Since MFCC avg, min and max are provided there is multiplication by three.
num_of_ceps = 13 * 3

# Number of training epochs.
nb_epochs = 1000

# Training batch size.
batch_size = 128

model = Sequential()


def compile_basic_model():
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    model.add(Dense(512, input_dim=num_of_ceps))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(512))
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='mae', optimizer=sgd, metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model_basic.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model_basic.json.')


def train_network():
    x, y = read_mfcc(DATASET_DIR)
    x_test, y_test = read_mfcc(TEST_DIR)
    print ('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print ('\nTraining network started\n')

    normalize_cepstrum_coefficients(x)
    normalize_cepstrum_coefficients(x_test)

    checkpointer = ModelCheckpoint(filepath="..\..\data\weights_checkpoint\weights_basic_best_acc.hdf5", verbose=1,
                                   save_best_only=True, monitor='val_acc', mode='auto')
    model.fit(x, y, nb_epoch=nb_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[checkpointer])
    model.save_weights('..\..\data\weights_basic.h5', overwrite=True)
    score = model.evaluate(x_test, y_test)
    print ('\nNetwork trained successfully and network weights saved as file weights_basic.h5.')
    print('\nTest score:', score[0])
    print('\nTest accuracy:', score[1])


def read_mfcc(data_dir):
    x = []
    y = []
    for label, genre in enumerate(GENRES_ALL):
        if genre not in GENRES:
            continue
        for file in glob.glob(os.path.join(data_dir, genre, "*.ceps.npy")):
            print ('Extracting MFCC from ' + file)
            ceps = np.load(file)
            num_ceps = len(ceps)
            temp_signal = []
            temp_signal.extend(np.mean(ceps[0:num_ceps], axis=0))
            temp_signal.extend(np.min(ceps[0:num_ceps], axis=0))
            temp_signal.extend(np.max(ceps[0:num_ceps], axis=0))
            x.append(temp_signal)
            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(x), np.array(y)


def normalize_cepstrum_coefficients(x):
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    return x

if __name__ == "__main__":
    compile_basic_model()
    train_network()
