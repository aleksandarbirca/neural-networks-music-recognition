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
import matplotlib.pyplot as plt


class DialogWindow(QtGui.QWidget):
    model = Sequential()

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.resize(640, 240)
        self.button = QtGui.QPushButton('Load .mp3 file', self)
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
            print "Format not supported."
            return

        song = AudioSegment.from_file(filename, file_format)
        song = song[:30000]
        wav_file = song.export(filename[:-format_length] + "_temp.wav", format='wav').name
        sample_rate, data = scipy.io.wavfile.read(wav_file)
        data[data == 0] = 1
        os.remove(wav_file)
        print "Removed temporary file: " + wav_file

        ceps, mspec, spec = mfcc(data)
        num_ceps = len(ceps)
        X = []
        X.append(np.mean(ceps[0:num_ceps], axis=0))
        X = np.array(X)
        X = scale(X, axis=1, with_mean=True, with_std=True, copy=True)
        model = model_from_json(open('..\data\weights\model.json','r').read())
        model.load_weights('..\data\weights\weights.h5')

        Y = model.predict(X)
        self.textbox.clear()
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
        print Y
        x = np.array(Y[0])
        print x
        self.draw_bar(x)

    def draw_bar(self, x):
        n_genres = 10
        index = np.arange(n_genres)
        bar_width = 0.95
        #        x=[random.uniform(0.1,1) for _ in range (10)]
        x_round = []
        fig, ax = plt.subplots()
        for i in x:
            x_round.append(float("{0:.2f}".format(self.loop_func(i))))
        rec = ax.bar(index, x_round, bar_width, color=[np.random.rand(3, 1) for _ in range(10)])
        ax.set_title('')
        ax.set_xticks(index + .3)
        ax.set_xticklabels(
            ('Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'))
        # ax.legend(rec,('Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'))
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


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    window = DialogWindow()
    window.show()
    sys.exit(app.exec_())
