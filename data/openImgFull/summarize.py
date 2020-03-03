import csv
import random
from os import listdir
from os.path import isfile, join

users = []
total = []
keys = []

with open('sampleInfo', 'r') as fin:
    allLines = fin.readlines()
    keys = [x for x in allLines[0].strip().split()]

    lines = allLines[1:]

    total = [int(x) for x in lines[0].strip().split()]
    users.append(total)

    for line in lines[1:]:
        instances = [int(x) for x in line.strip().split()]

        for i, item in enumerate(instances):
            total[i] += item
        users.append(instances)

indice = sorted(range(len(total)), key=lambda k: total[k], reverse=True)
sortedUserIndice = sorted(range(len(users)), key=lambda k: sum(users[k]), reverse=True)

with open('clientSamples', 'w') as fout:
    for userId in sortedUserIndice:
        fout.writelines(str(sum(users[userId])) + '\n')

with open('clientInfo', 'w') as fout:
    fout.writelines('\t'.join([keys[x] for x in indice]) + '\n')

    for userId in sortedUserIndice:
        fout.writelines('\t'.join([str(users[userId][x]) for x in indice]) + '\n')


