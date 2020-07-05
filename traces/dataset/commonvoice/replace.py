import os, csv


file_name = 'train_manifest.csv'

csvFile = open(file_name, 'r')
reader = csv.reader(csvFile)

results = []

replace = '/mnt/voice/english/train/'
replace_to = '/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/voice/train2/'

for item in reader:
    #print(item[0], '#', item[1])
    results.append([item[0].replace(replace, replace_to), item[1].replace(replace, replace_to)])

with open("train_manifest_16000.csv","w") as csvfile: 
    writer = csv.writer(csvfile, delimiter=',')

    writer.writerows(results)

