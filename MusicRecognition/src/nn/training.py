from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding
import os
import glob
import numpy as np
from keras.optimizers import SGD
from sklearn.preprocessing import scale

config = {}
execfile("..\..\config.cfg", config)
GENRE_LIST = config["GENRE_LIST"]
DATASET_DIR = config["DATASET_DIR"]

model = Sequential()

def compile_model():
    sgd = SGD(lr=0.9, decay=1e-6, momentum=0.8, nesterov=True)
    model.add(Embedding(100, 100))
    model.add(LSTM(13, return_sequences=True, input_shape=(300000, 13)))
    model.add(LSTM(13, return_sequences=True))
    model.add(LSTM(13))
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.1))
    #model.add(Dense(128))
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model.json', 'w').write(json_model)
    print '\nNetwork created successfully and network model saved as file model.json.'

def train_network():
    X, Y = read_mfcc()
    print '\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print '\nTraining network started\n'
    X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )
    # Y = scale( Y, axis=0, with_mean=True, with_std=True, copy=True )
    #X = (X - np.min(X)) / (np.max(X) - np.min(X))
    #Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    model.fit(X, Y, nb_epoch=1000, batch_size=128,  validation_data=(X, Y))
    model.save_weights('..\..\data\weights.h5', overwrite=True)
    score = model.evaluate(X, Y)
    print 'Network trained successfully and network weights saved as file weights.h5.'
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def read_mfcc():
    X = []
    y = []
    for label, genre in enumerate(GENRE_LIST):
        for file in glob.glob(os.path.join(DATASET_DIR, genre, "*.ceps.npy")):
            print 'Extracting MFCC from ' + file
            ceps = np.load(file)
            num_ceps = len(ceps)
            X.append(np.mean(ceps[0:num_ceps], axis=0))
            g = np.zeros(10)
            g[label.real] = 1
            y.append(g)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    compile_model()
    train_network()