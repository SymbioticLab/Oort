import pickle
import gc

gc.disable()

file_name = '/gpfs/gpfs0/groups/chowdhury/fanlai/dataset/clientprofile'
with open(file_name, 'rb') as fin:
    client_profile = pickle.load(fin)

def generate(client_list, batch_size, upload_epoch, model_size):
    completion_time = []

    for item in client_list:
        completion_time.append(4.0 * batch_size * upload_epoch*float(client_profile[item][0])/1000. + model_size/float(client_profile[item][1]))

    return completion_time

client_list = list(client_profile.keys())
completion_time = generate(client_list, batch_size=16, upload_epoch=10, model_size=30 * 8 * 1000)

with open('completion_time.pkl', 'wb') as fout:
    pickle.dump(completion_time, fout, -1)

print(completion_time[:100])
