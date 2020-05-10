import csv
import random
from os import listdir
from os.path import isfile, join

random.seed(233)

imgIdToAuthor = {}

# Load img csv
with open('train-images-boxable-with-rotation.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')

    for row in csv_reader:
        imgId = row[0]
        author = row[6]

        imgIdToAuthor[imgId] = author


authorToObjs = {}
categories = set([])

filePrefix = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']
for prefix in filePrefix:
    folder = './crop/crop' + prefix
    imgFiles = ([f for f in listdir(folder) if isfile(join(folder + '/', f)) and '.jpg' in f])

    print('Now working on file crop_{}'.format(prefix))

    for img in imgFiles:
        items = img.replace('.jpg', '').split('__')

        imgId = items[0]
        category = items[1]
        author = imgIdToAuthor[imgId]

        if category not in categories:
            categories.add(category)

        if author not in authorToObjs:
            authorToObjs[author] = {category: 1}
        else:
            cnt = authorToObjs[author].get(category, 0)
            cnt += 1
            authorToObjs[author][category] = cnt


keys = sorted(authorToObjs.keys())
objKeys = sorted(categories)

sampleCollection = []

for author in authorToObjs:
    sampleInfo = ''
    for key in objKeys:
        cnt = authorToObjs[author].get(key, 0)
        sampleInfo = sampleInfo + '\t' + str(cnt)
    sampleCollection.append(sampleInfo + '\n')

with open('sampleInfo', 'w') as fout:
    # write the head file
    fout.writelines('\t'.join([str(x) for x in objKeys]) + '\n')

    for info in sampleCollection:
        fout.writelines(info)
