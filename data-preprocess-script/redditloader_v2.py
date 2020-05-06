import json, os, gc, re
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
import operator
import pickle

def parse_clients(filename):
    with open(filename, 'r') as f:
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

            # Clean up text
            dataBody = re.sub(r'[^\w\s]','', dataBody)
            dataBody = re.sub('<[^<]+?>', '', dataBody)
            tokens = word_tokenize(dataBody)
            clientLen[curClientId] += len(tokens)
            clientCnt[curClientId] += 1

            if curClientId not in tempDic:
                tempDic[curClientId] = tokens
            else:
                tempDic[curClientId].extend(tokens)

            cnt += 1
            
            if cnt % 500000 == 0:
                print ('current ... {}'.format(cnt))

                # Dump 
                filename = "./RC_201712/" + filename + "_" + str(cnt)
                outfile = open(filename,'wb')
                pickle.dump(tempDic, outfile)
                outfile.close()

                del tempDic
                tempDic = {}
                gc.collect()
            
            clientLen_out = "./RC_201712/clientLen"
            outfile = open(clientLen_out,'wb')
            pickle.dump(clientLen, outfile)
            outfile.close()

            # break

parse_clients('RC_2017-12')