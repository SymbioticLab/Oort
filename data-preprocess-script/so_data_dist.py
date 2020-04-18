import h5py as h5
import pickle
print("Running")
filename1 = "vocab_tokens.txt"
with open(filename1, 'rb') as f:
    top_tokens = pickle.load(f)[:10000]

token_dict = {k: v for v, k in enumerate(top_tokens)}

f = h5.File("stackoverflow_train.h5", "r")
# for i in f['examples']['10805230']['tags']:
    # print(i)

client_dict = {}

count = 0
client_list = list(f['examples'])
for client in client_list:
    titles_list = list(f['examples'][client]['title'])
    tokens_list = list(f['examples'][client]['tokens'])
    client_dict[client] = [0] * 10000
    count += 1
    for title, token in zip(titles_list, tokens_list):
        str_list = title.decode("UTF-8").split() + token.decode("UTF-8").split()
        for s in str_list:
            if s.lower() not in token_dict:
                continue
            client_dict[client][token_dict[s.lower()]] += 1

print(len(client_dict))

filename = "so_data_size"
outfile = open(filename, 'wb')
pickle.dump(client_dict, outfile)
outfile.close()

"""
#print(type(list(f['examples']['00000001']['tokens'])[0].decode('UTF-8')))
#count = 0
#for client in client_list:
#    count += len(list(f['examples'][client]['tokens']))
#print(count)
# for i in f['examples']['00000001']['tokens']:
#     print(i)
#     count += 1

# print(count)
# print(f['examples']['10805230']['tags'])
# for i in f['examples']['10805230]:
#     print(i)
"""
