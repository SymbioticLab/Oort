import json, os, gc, re
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
import operator
import pickle

class clientInfo():
    def __init__(self):
        self.uniqueIdToAuthor = {}
        self.idToName = {}
        self.clientCnt = {}
        self.clientLen = {}
        self.clientId = 0


def parse_clients(infilename, flag, dirname):
    print("started")
    with open(infilename, 'r') as f:

        cnt = 1

        if flag:
            clientInfo_dict = clientInfo()
        else:
            infile = "clientInfo"
            infile = open(infile, 'rb')
            clientInfo_dict = pickle.load(infile)
        tempDic = {}


        for line in f:
            data = json.loads(line)

            if data['author'] not in clientInfo_dict.uniqueIdToAuthor:
                clientInfo_dict.uniqueIdToAuthor[data['author']] = clientInfo_dict.clientId
                clientInfo_dict.idToName[clientInfo_dict.clientId] = str(data['author'].encode('ascii', 'ignore').decode('ascii'))
                clientInfo_dict.clientCnt[clientInfo_dict.clientId] = 0
                clientInfo_dict.clientLen[clientInfo_dict.clientId] = 0

                clientInfo_dict.clientId += 1

            dataBody = str(data['body'].encode('ascii', 'ignore').decode('ascii')).strip()
            curClientId = clientInfo_dict.uniqueIdToAuthor[data['author']]

            # Clean up text
            dataBody = re.sub(r'[^\w\s]','', dataBody)
            dataBody = re.sub('<[^<]+?>', '', dataBody)
            tokens = word_tokenize(dataBody)
            clientInfo_dict.clientLen[curClientId] += len(tokens)
            clientInfo_dict.clientCnt[curClientId] += 1

            if curClientId not in tempDic:
                tempDic[curClientId] = tokens
            else:
                tempDic[curClientId].extend(tokens)


            cnt += 1

            if cnt % 1000 == 0:
                print ('current ... {}'.format(cnt))

                # Dump
                filename = dirname + infilename + "_" + str(cnt)

                outfile = open(filename,'wb')
                pickle.dump(tempDic, outfile)
                outfile.close()

                del tempDic
                tempDic = {}
                gc.collect()
                #break

            clientInfo_out = 'clientInfo'
            outfile = open(clientInfo_out, 'wb')
            pickle.dump(clientInfo_dict, outfile)
            outfile.close()




parse_clients('RC_2017-12', True, "./RC_201712/")
parse_clients('RC_2018-01', False, "./RC_201801/")
parse_clients('RC_2018-02', False, "./RC_201802/")