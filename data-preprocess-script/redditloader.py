import json, os, gc
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
import operator

def parse_clients():
    with open('RC_2019-08', 'r') as f:
        uniqueIdToAuthor = {}
        idToName = {}
        clientId = 0

        cnt = 1

        tempDic = {}
        clientCnt = {}
        clientLen = {}

        for line in f:
            data = json.loads(line)

            if data['author'] not in uniqueIdToAuthor:
                uniqueIdToAuthor[data['author']] = clientId
                idToName[clientId] = str(data['author'].encode('ascii', 'ignore').decode('ascii'))
                clientCnt[clientId] = 0
                clientLen[clientId] = 0

                clientId += 1

            dataBody = str(data['body'].encode('ascii', 'ignore').decode('ascii')).strip()
            curClientId = uniqueIdToAuthor[data['author']]

            clientCnt[curClientId] += 1
            clientLen[curClientId] += len(dataBody)

            if curClientId not in tempDic:
                tempDic[curClientId] = [dataBody]
            else:
                tempDic[curClientId].append(dataBody)

            cnt += 1
            # dump to file every N iters
            if cnt % 10000 == 0:
                print ('current ... {}'.format(cnt))
                # dump
                for client in tempDic.keys():
                    clientPath = './data/' + str(client)

                    if not os.path.exists(clientPath):
                        with open(clientPath, 'w') as f:
                            f.writelines(idToName[client] + '\n')
                            pass

                    with open(clientPath, 'a') as fout:
                        fout.writelines('\n'.join(tempDic[client]) + '\n')

                del tempDic

                tempDic = {}

                gc.collect()
                #break

    clientIds = sorted(clientCnt.keys())
    with open('./data/numSamples', 'w') as fout:
        fout.writelines('\n'.join([str(clientCnt[clientId]) for clientId in clientIds]))

    with open('./data/lenSamples', 'w') as fout:
        fout.writelines('\n'.join([str(clientLen[clientId]) for clientId in clientIds]))

def rm_redundancy():
    # load all users
    mypath = './data/'
    onlyfiles = [int(f) for f in listdir(mypath) if isfile(join(mypath, f))]

    rmCopy = 100000

    for i in range(rmCopy):
        newLines = []
        with open(join(mypath, str(i)), 'r') as fin:
            lines = fin.readlines()

            for line in lines:
                if line not in newLines:
                    newLines.append(line)

        with open(join(mypath, str(i)), 'w') as fout:
            fout.writelines(''.join(newLines))

dict = {}

def parse_words():
    mypath = './data/'
    onlyfiles = [int(f) for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        with open(join(mypath, str(file)), 'r') as fin:
            lines = fin.readlines()
            clientDic = {}

            for line in lines:
                line = line.strip()
                line = re.sub(r'[^\w\s]','', line)
                line = re.sub('<[^<]+?>', '', line)
                tokens = word_tokenize(line)

                for token in tokens:
                    count = dict.get(token.lower(), 0)
                    dict[token.lower()] = count + 1
                    clientcount = clientDic.get(token.lower(), 0)
                    clientDic[token.lower()] = clientcount + 1

            # dump for each client
            sorted_dict = sorted(clientDic.items(), key=operator.itemgetter(1), reverse=True)

            with open(join(mypath, str(file) + 'w'), 'w') as fout:
                for word in sorted_dict:
                    fout.writelines(word + '\t' + str(sorted_dict[word]) + '\n')

        # dump for all
        sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

        with open(join(mypath, 'wordSummary'), 'w') as fout:
            for word in sorted_dict:
                fout.writelines(word + '\t' + str(sorted_dict[word]) + '\n')

#rm_redundancy()
parse_clients()


