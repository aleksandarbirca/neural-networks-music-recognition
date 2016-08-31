from keras.layers import Dense, Activation, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
import numpy as np
import glob
import os

config = {}
exec(open("..\..\config.cfg").read(), config)

GENRES = config["GENRES"]
GENRES_ALL = config["GENRES_ALL"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]

# Number of MFCC samples.
num_of_mfcc = 40

# Number of coefficients in each sample.
# Since MFCC avg, min, max and var are provided there is multiplication by four.
num_of_ceps = 13 * 4

# Number of training epochs.
nb_epochs = 100

# Training batch size.
batch_size = 16

# Learning rate.
lr = 0.0001

model = Sequential()


def compile_lstm_model():
    model.add(LSTM(512, return_sequences=True, input_shape=(num_of_mfcc, num_of_ceps)))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=lr)
    model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model_lstm_mfcc.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model_lstm_mfcc.json.')


def train_network():
    x, y = read_mfcc(DATASET_DIR)
    x_test, y_test = read_mfcc(TEST_DIR)

    normalize_cepstrum_coefficients(x)
    normalize_cepstrum_coefficients(x_test)

    x = np.reshape(x, (x.shape[0], x.shape[1], num_of_ceps))
    x_test = np.reshape(x_test, (x_test.shape[0], x.shape[1], num_of_ceps))

    print ('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print ('\nNetwork training started\n')
    checkpointer = ModelCheckpoint(filepath="..\..\data\weights_checkpoint\weights_lstm_best_acc.hdf5",verbose=1,
                                   save_best_only=True, monitor='val_acc', mode='auto')
    model.fit(x, y, nb_epoch=nb_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[checkpointer])
    model.save_weights('..\..\data\weights_lstm_mfcc.h5', overwrite=True)
    score = model.evaluate(x_test, y_test)
    print ('\nNetwork trained successfully and network weights saved as file weights_lstm_mfcc.h5.')
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
            sampled_ceps = []
            for i in range(0, len(ceps)/100):
                temp_ceps = []
                temp_ceps.extend(np.mean(ceps[i*100:i*100+100], axis=0))
                temp_ceps.extend(np.min(ceps[i*100:i*100+100], axis=0))
                temp_ceps.extend(np.max(ceps[i*100:i*100+100], axis=0))
                temp_ceps.extend(np.var(ceps[i*100:i*100+100], axis=0))
                sampled_ceps.append(temp_ceps)

            sampled_ceps = sampled_ceps[0:num_of_mfcc]
            x.append(np.array(sampled_ceps))
            # TODO Use np_utils.to_categorical
            y_temp = np.zeros(10)
            y_temp[label.real] = 1
            y.append(y_temp)
    return np.array(x), np.array(y)


def normalize_cepstrum_coefficients(x):
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    return x

if __name__ == "__main__":
    compile_lstm_model()
    train_network()
