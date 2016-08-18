from keras.models import Sequential
from keras.models import model_from_json
import scipy.io.wavfile
from PyQt4 import QtGui
import scipy.io.wavfile
import scipy
import numpy as np
from pydub import AudioSegment
from scikits.talkbox.features import mfcc
import os
from sklearn.preprocessing import scale

class DialogWindow(QtGui.QWidget):

    model = Sequential()

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.resize(320, 240)
        self.button = QtGui.QPushButton('Load .mp3 file', self)
        self.button.clicked.connect(self.handleButton)

        # Create textbox
        self.textbox = QtGui.QTextEdit(self)
        self.textbox.setReadOnly(True)

        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(self.textbox)

    def handleButton(self):
        # Get filename using QFileDialog
        self.textbox.clear()
        
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '/')
        print 'Opened file ' + filename
        
        filename = str(filename)
        song = AudioSegment.from_file(filename, "mp3")
        song = song[:30000]
        wav_file = song.export(filename[:-3] + "wav", format='wav').name
        sample_rate, data = scipy.io.wavfile.read(wav_file)
        data[data == 0] = 1
        os.remove(wav_file)
        ceps, mspec, spec = mfcc(data)
        num_ceps = len(ceps)
        X = []
        X.append(np.mean(ceps[0:num_ceps], axis=0))
        X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)
        model = model_from_json(open('..\data\weights\model.json', 'r').read())
        model.load_weights('..\data\weights\weights.h5')

        Y = model.predict(np.array(X))
        self.textbox.insertPlainText('Blues: ' + str(Y[0][0]))
        self.textbox.insertPlainText('\nClassical: ' + str(Y[0][1]))
        self.textbox.insertPlainText('\nCountry: ' + str(Y[0][2]))
        self.textbox.insertPlainText('\nDisco: ' + str(Y[0][3]))
        self.textbox.insertPlainText('\nHipHop: ' + str(Y[0][4]))
        self.textbox.insertPlainText('\nJazz: ' + str(Y[0][5]))
        self.textbox.insertPlainText('\nMetal: ' + str(Y[0][6]))
        self.textbox.insertPlainText('\nPop: ' + str(Y[0][7]))
        self.textbox.insertPlainText('\nReggae: ' + str(Y[0][8]))
        self.textbox.insertPlainText('\nRock: ' + str(Y[0][9]))

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    window = DialogWindow()
    window.show()
    sys.exit(app.exec_())