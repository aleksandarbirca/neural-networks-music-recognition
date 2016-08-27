import os
import timeit
import scipy.io.wavfile
import scipy
import numpy as np
from pydub import AudioSegment
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt


config = {}
execfile("..\..\config.cfg", config)

DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]
GENRE_LIST = config["GENRE_LIST"]


def convert_dataset_to_wav(data_dir):
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("au"):
                print 'Converting' + path + ' to wav.'
                song = AudioSegment.from_file(path, "au")
                #TODO Remove
                song = song[:30000]
                song.export(path[:-2]+"wav", format='wav')

    stop = timeit.default_timer()
    print "Files converted successfully."
    print "Conversion time = ", (stop - start)


# Mel Frequency Cepstral Coefficients
def convert_wav_to_mfcc(data_dir, ceps_num):
    start = timeit.default_timer()
    print "Starting conversion to MFCC."
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("wav"):
                extract_cepstrum(path, ceps_num)

    stop = timeit.default_timer()
    print "Files converted successfully."
    print "Conversion time = ", (stop - start)


def extract_cepstrum(path, ceps_num):
    sample_rate, signal = scipy.io.wavfile.read(path)
    signal[signal == 0] = 1
    #plt.plot(signal)
    #signal=signal[0::1000]
    #plt.specgram(signal)
    #plt.show()
    ceps, mspec, spec = mfcc(signal, nceps=ceps_num)
    base, ext = os.path.splitext(path)
    data = base + ".ceps"
    np.save(data, ceps)
    print "Written ", data


if __name__ == "__main__":
    #convert_dataset_to_wav(DATASET_DIR)
    #convert_dataset_to_wav(TEST_DIR)
    convert_wav_to_mfcc(DATASET_DIR, 13) #Number of MFCC (13-40)
    convert_wav_to_mfcc(TEST_DIR, 13)
