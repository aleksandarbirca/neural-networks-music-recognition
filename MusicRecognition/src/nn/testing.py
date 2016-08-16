from keras.models import Sequential
import os
import numpy as np
from sklearn.preprocessing import scale
from keras.models import model_from_json
from keras.optimizers import SGD
import timeit
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
from PyQt4 import QtGui


class DialogWindow(QtGui.QWidget):
    config = {}
#    execfile("..\..\config.cfg", config)
#    GENRE_LIST = config["GENRE_LIST"]

    model = Sequential()

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.resize(320, 240)
        self.button = QtGui.QPushButton('Load wav file...', self)
        self.button.clicked.connect(self.handleButton)

        # Create textbox
        self.textbox = QtGui.QTextEdit(self)
        self.textbox.setReadOnly(True)

        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(self.textbox)

    def handleButton(self):
        # Get filename using QFileDialog
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '/')
        print filename

        # print file contents
#        with open(filename, 'r') as f:
        filepath = self.convert_wav_to_mfcc(filename)
        self.test_network(filepath)

        # Mel Frequency Cepstral Coefficients

    def convert_wav_to_mfcc(self, DATASET_DIR):
        start = timeit.default_timer()
        print "Starting conversion to MFCC."
        self.extract_cepstrum(DATASET_DIR)  # Mel-frequency cepstrum (https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

        stop = timeit.default_timer()
        print "Files converted successfully."
        print "Conversion time = ", (stop - start)
        return DATASET_DIR

    def extract_cepstrum(self,path):
        sample_rate, X = scipy.io.wavfile.read(path)
        X[X == 0] = 1
        ceps, mspec, spec = mfcc(X)
        base_fn, ext = os.path.splitext(path)
        data_fn = base_fn + ".ceps"
        np.save(data_fn, ceps)
        print "Written ", data_fn

    def test_network(self, TEST_DIR):
        print '\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print '\nTesting network started\n'
        # load json and create model
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '/')
        json_file = open(filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        print loaded_model_json
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights('..\..\data\weights.h5')
        print("Loaded model from disk")

        X = self.read_mfcc(TEST_DIR)
        X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)
        print X
        # prediction loaded model on test data
        sgd = SGD(lr=0.9, decay=1e-6, momentum=0.8, nesterov=True)
        loaded_model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
        y_pred = self.model.predict(X)
        print(y_pred)
        self.textbox.insertPlainText(y_pred)

    def read_mfcc(self, TEST_DIR):
        print TEST_DIR
        X = []
        #        for label, genre in enumerate(self.GENRE_LIST):
        #            for fn in glob.glob(os.path.join(TEST_DIR, genre, "*.ceps.npy")):
        print 'Extracting MFCC from ' + TEST_DIR
        ceps = np.load(TEST_DIR)
        num_ceps = len(ceps)
        X.append(np.mean(ceps[0:num_ceps], axis=0))  # mean value from 4000+ CEPS samples
        return np.array(X)


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    window = DialogWindow()
    window.show()
    sys.exit(app.exec_())

