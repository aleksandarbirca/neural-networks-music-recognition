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
import matplotlib.pyplot as plt


class DialogWindow(QtGui.QWidget):
    model = Sequential()

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.resize(640, 240)
        self.setWindowTitle('Music Recognition')
        self.button = QtGui.QPushButton('Load audio file', self)
        self.button.setFixedWidth(100)
        self.button.clicked.connect(self.handle_button)

        # Create textbox
        self.textbox = QtGui.QTextEdit(self)
        self.textbox.setReadOnly(True)

        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(self.textbox)

    def handle_button(self):
        # Get filename using QFileDialog
        self.textbox.clear()

        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '/')
        self.textbox.insertPlainText('Opened file ' + filename)
        print 'Opened file ' + filename
        filename = str(filename)

        if filename.endswith("au"):
            file_format = "au"
            format_length = 3
        elif filename.endswith("mp3"):
            file_format = "mp3"
            format_length = 4
        elif filename.endswith("wav"):
            file_format = "wav"
            format_length = 4
        else:
            self.textbox.insertPlainText("\nFormat not supported.")
            print 'Format not supported.'
            return

        song = AudioSegment.from_file(filename, file_format)
        song = song[:30000]
        wav_file = song.export(filename[:-format_length] + "_temp.wav", format='wav').name
        sample_rate, signal = scipy.io.wavfile.read(wav_file)
        signal[signal == 0] = 1
        os.remove(wav_file)
        self.textbox.insertPlainText("\nRemoved temporary file: " + wav_file)
        print 'Removed temporary file: ' + wav_file

        #TODO set sample rate depending on signal length and remove song = song[:30000]
        self.textbox.insertPlainText('\nExtracting MFCC from file.')
        self.textbox.updatesEnabled()
        ceps, mspec, spec = mfcc(signal)
        num_ceps = len(ceps)
        x = [np.mean(ceps[0:num_ceps], axis=0)]
        x = np.array(x)
        x = normalize_cepstrum_coefficients(x)
        self.textbox.insertPlainText('\nLoading model and weights from ..\data\weights folder.')
        print 'Loading model and weights from ..\data\weights folder.'
        model = model_from_json(open('..\data\weights\model.json', 'r').read())
        model.load_weights('..\data\weights\weights.h5')

        self.textbox.insertPlainText('\nDeterminating genre.')
        print 'Determinating genre.'
        y = model.predict(x)
        x = np.array(y[0])
        self.draw_bar(x)
        self.textbox.insertPlainText('\nDone!')
        print 'Done!'

    def draw_bar(self, x):
        n_genres = 10
        index = np.arange(n_genres)
        bar_width = 0.95
        x_round = []
        fig, ax = plt.subplots()
        for i in x:
            x_round.append(float("{0:.2f}".format(self.loop_func(i))))
        rec = ax.bar(index, x_round, bar_width, color=[np.random.rand(3, 1) for _ in range(10)])
        ax.set_title('')
        ax.set_xticks(index + .3)
        ax.set_xticklabels(
            ('Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'))
        ax.set_xlabel('Genres')
        ax.set_ylabel('Accuracy')
        self.autolabel(rec, x)
        plt.show()

    @staticmethod
    def loop_func(i):
        numb = 0
        for j in range(0, 20):
            numb += 0.05
            if np.arange(0, numb, i).any():
                return numb

    @staticmethod
    def autolabel(rects, x):
        i = 0
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                     '%f' % float(x[i]),
                     ha='center', va='bottom')
            i += 1


def normalize_cepstrum_coefficients(x):
    x[:, 0] = x[:, 0] / 20
    x[:, 1:13] = (x[:, 1:13] + 3) / 7
    return x

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    window = DialogWindow()
    window.show()
    sys.exit(app.exec_())
