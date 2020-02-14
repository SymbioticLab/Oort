import os
import pickle
import re
import operator
from nltk import word_tokenize
from data_div import measureAllDistance

"""
dict = {}

def parse_email(sentence):
    for sentence in sentence:
        sentence = re.sub(r'[^\w\s]','',sentence)
        tokens = word_tokenize(sentence)
        for token in tokens:
            count = dict.get(token.lower(), 0)
            dict[token.lower()] = count + 1

folders = os.listdir('maildir')



# Iterate all names
for name in folders:
    sent_folder = './maildir/' + name + '/sent/'
    emails = os.listdir(sent_folder)
    print(sent_folder)

    # Process all emails
    for email in emails:
        print(sent_folder + ' ' +email)
        with open(sent_folder + email, 'r') as f:
                lines = f.read().split('\n\n')[1:]
                parse_email(lines)

# Save the count dict
filename = "count.txt"
outfile = open(filename,'wb')
pickle.dump(dict, outfile)
outfile.close()



# Load the dict from disk
filename = "count.txt"
infile = open(filename,'rb')
dict = pickle.load(infile)
infile.close()

sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)[:5000]
top_words = [word[0] for word in sorted_dict]



filename = "top_words.txt"
outfile = open(filename,'wb')
pickle.dump(top_words, outfile)
outfile.close()


filename = "top_words.txt"
infile = open(filename, 'rb')
top_words = pickle.load(infile)
infile.close()

print(top_words[:20])




# person_to_words = {}
# person_to_number_of_samples = [0] * 150

data_dist = [[0] * 5000 for _ in range(150)]
index = -1

def parse_top_words(sentence, name, index):
    for sentence in sentence:
        sentence = re.sub(r'[^\w\s]','',sentence)
        tokens = word_tokenize(sentence)
        for token in tokens:
            if token.lower() not in top_words:
                continue
            data_dist[index][top_words.index(token.lower())] += 1

folders = os.listdir('maildir')

# Iterate all names
for name in folders:
    sent_folder = './maildir/' + name + '/sent/'
    emails = os.listdir(sent_folder)
    print(sent_folder)
    index += 1
    # Process all emails
    for email in emails:
        print(sent_folder + email)
        with open(sent_folder + email, 'r') as f:
                lines = f.read().split('\n\n')[1:]
                parse_top_words(lines, name, index)


filename = "email_data_dist.txt"
outfile = open(filename,'wb')
pickle.dump(data_dist, outfile)
outfile.close()
"""





filename = "email_data_dist.txt"
infile = open(filename,'rb')
dict = pickle.load(infile)
infile.close()

data_size = [sum(i) for i in dict]

print(data_size)

measureAllDistance(dict)
