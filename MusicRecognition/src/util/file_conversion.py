import os
import timeit
import scipy.io.wavfile
import scipy
import numpy as np
from pydub import AudioSegment
from scikits.talkbox.features import mfcc


config = {}
execfile("..\..\config.cfg", config)

DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]
GENRE_LIST = config["GENRE_LIST"]


def convert_dataset_to_wav():
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("au"):
                print 'Converting: ' + path + ' to wav.'
                song = AudioSegment.from_file(path, "au")
                song = song[:30000]
                song.export(path[:-2]+"wav", format='wav')

    stop = timeit.default_timer()
    print "Files converted successfully."
    print "Conversion time = ", (stop - start)


# Mel Frequency Cepstral Coefficients
def convert_wav_to_mfcc():
    start = timeit.default_timer()
    print "Starting conversion to MFCC."
    for subdir, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("wav"):
                extract_cepstrum(path)  # Mel-frequency cepstrum (https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

    stop = timeit.default_timer()
    print "Files converted successfully."
    print "Conversion time = ", (stop - start)


def extract_cepstrum(path):
    sample_rate, x = scipy.io.wavfile.read(path)
    x[x == 0] = 1
    ceps, mspec, spec = mfcc(x)
    base_fn, ext = os.path.splitext(path)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print "Written ", data_fn


if __name__ == "__main__":
    convert_dataset_to_wav()
    convert_wav_to_mfcc()