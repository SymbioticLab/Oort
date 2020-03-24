import csv
from collections import OrderedDict
import numpy as np 
import pickle
import time

def read_img_labels():
    imgToLabels = OrderedDict()
    categories = OrderedDict()
    categoryId = 0

    with open('extended-crowdsourced-image-labels.csv', 'r') as csvfile:
        imgLabels = csv.reader(csvfile, delimiter=',')
        next(imgLabels)

        for row in imgLabels:
            #row = row.split(',')
            imgId, label = row[0].strip(), row[2].strip()
            imgToLabels[imgId] = label

            if label not in categories:
                categories[label] = categoryId
                categoryId += 1

    return imgToLabels, categories

def read_img_clients():
    clientToImg = OrderedDict()

    with open('extended-crowdsourced-image-ids.csv', 'r') as csvfile:
        imgLabels = csv.reader(csvfile, delimiter=',')
        next(imgLabels)

        for row in imgLabels:
            #row = row.split(',')
            imgId, client = row[0].strip(), row[6].strip()

            if client not in clientToImg:
                clientToImg[client] = []

            clientToImg[client].append(imgId)

    return clientToImg

def sort_data():
    with open('blog_data_dist.txt', 'rb') as fin:
        temp = pickle.load(fin)

    nclients = len(temp)
    nclass = len(temp[0])

    print('nclass: {}, nclients: {}'.format(nclass, nclients))
    
    clientClass = np.zeros([nclients, nclass])

    stime = time.time()
    for row in range(nclients):
        for col in range(nclass):
            clientClass[row][col] = temp[row][col]

    print('assign clientClass takes {} s'.format(time.time() - stime))

    stime = time.time()
    rowSums = [sum(clientClass[i]) for i in range(nclients)]
    colSums = [sum(clientClass[:, i]) for i in range(nclass)]

    rowIndex = sorted(range(len(rowSums)), key=lambda k: rowSums[k], reverse=True)
    colIndex = sorted(range(len(colSums)), key=lambda k: colSums[k], reverse=True)

    print('Sort clientClass takes {} s'.format(time.time() - stime))
    stime = time.time()
    # copy to a new matrix
    tempClientClass = np.zeros([nclients, nclass])

    for i, item in enumerate(rowIndex):
        for k, itemc in enumerate(colIndex):
            tempClientClass[i][k] = clientClass[item][itemc]

    print('Assign tempClientClass takes {} s'.format(time.time() - stime))

    with open('blog_data_dist_sort.txt', 'wb') as fout:
        pickle.dump(tempClientClass, fout, protocol=4)

def analyze():
    imgToLabels, categories = read_img_labels()
    clientToImg = read_img_clients()
    clientDict = OrderedDict()
    nclass = len(categories)
    nclients = len(clientToImg)

    print('nclass: {}, nclients: {}'.format(nclass, nclients))
    
    clientClass = np.zeros([nclients, nclass])

    for Id, client in enumerate(clientToImg):
        for img in clientToImg[client]:
            labelID = categories[imgToLabels[img]]
            clientClass[Id][labelID] += 1

    rowSums = [sum(clientClass[i]) for i in range(nclients)]
    colSums = [sum(clientClass[:, i]) for i in range(nclass)]

    rowIndex = sorted(range(len(rowSums)), key=lambda k: rowSums[k], reverse=True)
    colIndex = sorted(range(len(colSums)), key=lambda k: colSums[k], reverse=True)

    # copy to a new matrix
    tempClientClass = np.zeros([nclients, nclass])

    for i, item in enumerate(rowIndex):
        for k, itemc in enumerate(colIndex):
            tempClientClass[i][k] = clientClass[item][itemc]

    # # sort clients
    # for i in range(nclients):
    #     for k in range(i + 1, nclients):
    #         if rowSums[i] < rowSums[k]:
    #             # swap
    #             clientClass[[i, k]] = clientClass[[k ,i]]

    # colSums = [sum(clientClass[:, i]) for i in range(nclass)]

    # # sort class 
    # for i in range(nclass):
    #     for k in range(i + 1, nclass):
    #         if colSums[i] < colSums[k]:
    #             # swap
    #             clientClass[:, [i, k]] = clientClass[:, [k ,i]]


    with open('clientSamples', 'wb') as fout:
        pickle.dump(tempClientClass, fout)

#analyze()
def load_data():
    with open('clientSamples', 'rb') as fin:
        clientClass = pickle.load(fin)

    nclients = clientClass.shape[0]
    nclass = clientClass.shape[1]

    for i in range(nclients):
        print(sum(clientClass[i]))

    print('==================')
    for i in range(nclass):
        print(sum(clientClass[:, i]))

#load_data()
sort_data()
