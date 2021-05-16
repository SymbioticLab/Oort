import logging
import threading
import time
import glob
import json
import pickle



def preprocess_reddit(filename):
    logging.info("Start Processing %s", filename)

    # Load client's data
    with open(filename, 'rb') as cin:
        data = json.load(cin)

    # Load top words
    with open('reddit_vocab.pck', 'rb') as vin:
        vocab = pickle.load(vin)['vocab']

    client_dict = {}

    for usr in data['users']:
        client_dict[usr] = [0] * len(vocab)

        for sent in data['user_data'][usr]['x']:
            for word in sent.split():
                if word not in vocab:
                    continue
                client_dict[usr][vocab[word]] += 1
    
    with open(filename + '.pkl', 'wb') as fout:
        pickle.dump(client_dict, fout)

    logging.info("Finish Processing %s", filename)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    path = "*.json"
    files = glob.glob(path)
    files.sort()


    for f in files[:1]:
        x = threading.Thread(target=preprocess_reddit, args=(f,))
        x.start()
