import os, csv, pickle

wav_prefix = '/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/voice/train2/wav/'
txt_prefix = '/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/voice/train2/txt/'

def concatenate_files(alist):
    results = []

    for wav in alist:
        results.append([wav_prefix+wav, txt_prefix+wav.replace('.wav', '.txt')])

    return results

# load testing file
with open('testWavs', 'rb') as fin:
    testing_wavs = pickle.load(fin)

with open('trainingWavs', 'rb') as fin:
    training_wavs = pickle.load(fin)

training_wavs = concatenate_files(list(training_wavs))
testing_wavs = concatenate_files(list(testing_wavs))

with open("train_manifest.csv","w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(training_wavs)

with open("test_manifest.csv","w") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(testing_wavs)
