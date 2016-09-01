from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os
import glob
import numpy as np

config = {}
exec(open("..\..\config.cfg").read(), config)
GENRES = config["GENRES"]
GENRES_ALL = config["GENRES_ALL"]
DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]

######################################################################################################################
# This is just initial implementation. The main purpose of this implementation is to see how CNN is going to perform #
# with image-like representation of non image data, in this case numerical representation of music spectrogram.      #
######################################################################################################################


# Number of MFCC samples.
rows = 80

# Number of coefficients in each sample.
columns = 13

# Convolutional kernel width.
kernel_width = 5

# Number of training epochs.
nb_epochs = 1000

# Training batch size.
batch_size = 64

model = Sequential()


def compile_convolution2d_model():

    model.add(Convolution2D(15, kernel_width, columns, border_mode='same', input_shape=(1, rows, columns)))
    model.add(Activation('relu'))
    model.add(Convolution2D(15, kernel_width, columns, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(15, kernel_width, columns, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mae', optimizer=sgd, metrics=['accuracy'])

    json_model = model.to_json()
    open('..\..\data\model_conv2d.json', 'w').write(json_model)
    print ('\nNetwork created successfully and network model saved as file model_conv2d.json.')


def train_network():
    x, y = read_mfcc(DATASET_DIR)
    x_test, y_test = read_mfcc(TEST_DIR)
    print ('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print ('\nTraining network started\n')

    normalize_cepstrum_coefficients(x)
    normalize_cepstrum_coefficients(x_test)

    # (number of samples x 1 channel x rows x columns)
    x = x.reshape(x.shape[0], 1, rows, columns)
    x_test = x_test.reshape(x_test.shape[0], 1, rows, columns)

    checkpointer = ModelCheckpoint(filepath="..\..\data\weights_checkpoint\weights_conv2d_best_acc.hdf5", verbose=1,
                                   save_best_only=True, monitor='val_acc', mode='auto')
    model.fit(x, y, nb_epoch=nb_epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[checkpointer])

    model.save_weights('..\..\data\weights_conv2d.h5', overwrite=True)
    score = model.evaluate(x_test, y_test)
    print ('\nNetwork trained successfully and network weights saved as file weights_conv2d.h5.')
    print('\nTest score:', score[0])
    print('\nTest accuracy:', score[1])


def read_mfcc(data_dir):
    x = []
    y = []
    sampling_step = 50
    for label, genre in enumerate(GENRES_ALL):
        if genre not in GENRES:
            continue
        for file in glob.glob(os.path.join(data_dir, genre, "*.ceps.npy")):
            print ('Extracting MFCC from ' + file)
            ceps = np.load(file)
            sampled_ceps = []
            for i in range(0, len(ceps)/sampling_step):
                temp_ceps = []
                temp_ceps.extend(np.mean(ceps[i*sampling_step:i*sampling_step+sampling_step], axis=0))
                sampled_ceps.append(temp_ceps)

            sampled_ceps = sampled_ceps[0:rows]
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
    compile_convolution2d_model()
    train_network()
