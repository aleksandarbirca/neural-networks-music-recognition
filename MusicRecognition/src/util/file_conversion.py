from scikits.talkbox.features import mfcc
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import timeit
import os
import matplotlib.pyplot as plt

config = {}
execfile("..\..\config.cfg", config)

DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]
GENRES = config["GENRES_ALL"]

# Convert original audio files from dataset to WAV format
convert_to_wav = False

# Extract MFCC (Mel Frequency Cepstral Coefficients) from audio files and save them as .ceps files
extract_mfcc = True

# Number of MFCC (13-40)
mfcc_num = 13

# Save spectrogram image
save_spectrogram_image = True


def convert_dataset_to_wav(data_dir):
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("au"):
                print 'Converting' + path + ' to wav.'
                song = AudioSegment.from_file(path, "au")
                song = song[:30000]
                song.export(path[:-2]+"wav", format='wav')

    stop = timeit.default_timer()
    print "Files converted successfully."
    print "Conversion time = ", (stop - start)


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
    sample_rate, signal = wavfile.read(path)
    if save_spectrogram_image:
        plt.specgram(signal)
        plt.savefig(path[:-3]+"png")
        plt.close()
    signal[signal == 0] = 1
    ceps, mspec, spec = mfcc(signal, nceps=ceps_num)
    base, ext = os.path.splitext(path)
    data = base + ".ceps"
    np.save(data, ceps)
    print "Written ", data


if __name__ == "__main__":
    if convert_to_wav:
        convert_dataset_to_wav(DATASET_DIR)
        convert_dataset_to_wav(TEST_DIR)
    if extract_mfcc:
        convert_wav_to_mfcc(DATASET_DIR, mfcc_num)
        #convert_wav_to_mfcc(TEST_DIR, mfcc_num)
