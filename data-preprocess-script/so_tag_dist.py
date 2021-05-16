import h5py as h5
import pickle
filename1 = "vocab_tags.txt"
with open(filename1, 'rb') as f:
    top_tags = pickle.load(f)[:500]

tags_dict = {k: v for v, k in enumerate(top_tags)}
f = h5.File("stackoverflow_train.h5", "r")
print(list(f['examples']['00000001']))
for i in f['examples']['00000001']['tokens']:
    print(i)


"""
client_dict = {}

count = 0
client_list = list(f['examples'])
for client in client_list:
    tags_list = list(f['examples'][client]['tags'])
    client_dict[client] = [0] * 500
    count += 1
    print(count)
    for tag in tags_list:
        str_list = tag.decode("UTF-8").split("|")
        for s in str_list:
            if s.lower() not in tags_dict:
                continue
            client_dict[client][tags_dict[s.lower()]] += 1
filename = "so_tags_data_size"
outfile = open(filename, 'wb')
pickle.dump(client_dict, outfile)
outfile.close()
"""
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
