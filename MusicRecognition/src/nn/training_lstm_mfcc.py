from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop, Adam
import os
import numpy as np
import glob
from keras.callbacks import ModelCheckpoint


config = {}
exec(open("..\..\config.cfg").read(), config)
GENRE_LIST = config["GENRE_LIST"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]

model = Sequential()

num_of_mfcc = 40
num_of_ceps = 13 * 4


def compile_lstm_model():
    model.add(LSTM(512, return_sequences=True, input_shape=(num_of_mfcc, num_of_ceps)))
    model.add(Dropout(0.25))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.0001)
    #optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model_lstm_mfcc.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model.json.')


def train_network():
    x, y = read_mfcc(DATASET_DIR)
    x_test, y_test = read_mfcc(TEST_DIR)

    normalize_cepstrum_coefficients(x)
    normalize_cepstrum_coefficients(x_test)

    x = np.reshape(x, (x.shape[0], x.shape[1], num_of_ceps))
    x_test = np.reshape(x_test, (x_test.shape[0], x.shape[1], num_of_ceps))

    print ('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print ('\nNetwork training started\n')
    checkpointer = ModelCheckpoint(filepath="..\..\data\weights_temp_lstm_mfcc.hdf5", verbose=1, save_best_only=True,
                                   monitor='val_acc', mode='auto')
    model.fit(x, y, nb_epoch=1000, batch_size=16, validation_data=(x_test, y_test), callbacks=[checkpointer])
    model.save_weights('..\..\data\weights_lstm_mfcc.h5', overwrite=True)
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
            sampled_ceps = []
            for i in range(0, len(ceps)/100):
                temp = []
                temp.extend(np.mean(ceps[i*100:i*100+100], axis=0))
                temp.extend(np.min(ceps[i*100:i*100+100], axis=0))
                temp.extend(np.max(ceps[i*100:i*100+100], axis=0))
                temp.extend(np.var(ceps[i * 100:i * 100 + 100], axis=0))
                sampled_ceps.append(temp)

            sampled_ceps = sampled_ceps[0:num_of_mfcc]
            x.append(np.array(sampled_ceps))
            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(x), np.array(y)


def normalize_cepstrum_coefficients(x):
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)
    return x

if __name__ == "__main__":
    compile_lstm_model()
    train_network()
