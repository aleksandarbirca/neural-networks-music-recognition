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


class DialogWindow(QtGui.QWidget):
    config = {}
#    execfile("..\..\config.cfg", config)
#    GENRE_LIST = config["GENRE_LIST"]

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
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '/')
        print 'Opened file ' + filename

        filename = str(filename)
        song = AudioSegment.from_file(filename, "mp3")
        song = song[:30000]
        wav_file = song.export(filename[:-3] + "wav", format='wav').name
        sample_rate, data = scipy.io.wavfile.read(wav_file)
        os.remove(wav_file)
        ceps, mspec, spec = mfcc(data)
        num_ceps = len(ceps)
        X = []
        X.append(np.mean(ceps[0:num_ceps], axis=0))

        model = model_from_json(open('..\data\weights\model.json','r').read())
        model.load_weights('..\data\weights\weights.h5')

        Y = model.predict(np.array(X))
        print Y

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    window = DialogWindow()
    window.show()
    sys.exit(app.exec_())