import os
import timeit
from pydub import AudioSegment

config = {}
execfile("..\..\config.cfg", config)

DATASET_DIR = config["DATASET_DIR"]
TEST_DIR = config["TEST_DIR"]
GENRE_LIST = config["GENRE_LIST"]


def convert_dataset_to_wav():
    start = timeit.default_timer()
    rootdir = DATASET_DIR
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("au"):
                print 'Converting' + path + ' to wav.'
                song = AudioSegment.from_file(path,"au")
                song = song[:30000]
                song.export(path[:-2]+"wav",format='wav')

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = subdir+'/'+file
            if not path.endswith("wav"):
                os.remove(path)

    stop = timeit.default_timer()
    print "Files converted Successfully."
    print "Conversion time = ", (stop - start)


if __name__ == "__main__":
    convert_dataset_to_wav()