import os, csv, pickle

author_maps = {}
author_files = {}
author_ids = {}
idx = 0

path = '/mnt/voice/english/train_all/wav'
# load all file names
file_names = set(os.listdir(path))

total_samples = 0
def tsv_reader(db_type):
    global author_maps, author_files, total_samples

    file_name = db_type +  '.tsv'

    csvFile = open(file_name, 'r')
    reader = csv.reader(csvFile, delimiter=",")


    for item in reader:
        if reader.line_num == 1:
            continue

        item = item[0].split('\t')

        wav_name = item[1].replace('.mp3', '.wav')
        if wav_name not in file_names:
            continue

        author_name = item[0]
        words = item[2].split()

        if author_name not in author_maps:
          author_maps[author_name]  = 0
          author_files[author_name] = 0
          author_ids[author_name] = []

        author_maps[author_name] += len(words)
        author_files[author_name] += 1
        author_ids[author_name].append(wav_name)
        total_samples += 1

db_types=['validated', 'train', 'test', 'other']

for db_type in db_types:
    tsv_reader(db_type)

sorted_keys = sorted(list(author_files.keys()), key=lambda k:author_files[k], reverse=True)

with open('authorwords__' + 'all', 'w') as fout:
    for key in sorted_keys:
      fout.writelines(str(author_maps[key]) + '\t' + str(author_files[key]) + '\n')

author_keys = sorted(list(author_ids), key=lambda k:len(author_ids[k]), reverse=True)
cut_off = 0

train_samples = 0

file_to_author = {}
training_wavs = set()
for idx, author in enumerate(author_keys):
    if len(author_ids[author]) < 16:
        cut_off = idx
        break

    train_samples += len(author_ids[author])
    for wav in author_ids[author]:
        file_to_author[wav] = idx
        training_wavs.add(wav)

print('Training samples:', train_samples)
with open('wavToAuthor', 'wb') as fout:
    pickle.dump(file_to_author, fout)

testing_wavs = set()
for author in author_keys[cut_off:]:
    for wav in author_ids[author]:
        testing_wavs.add(wav)

print('Testing samples:', len(testing_wavs))
with open('testWavs', 'wb') as fout:
    pickle.dump(testing_wavs, fout)

with open('trainingWavs', 'wb') as fout:
    pickle.dump(training_wavs, fout)

