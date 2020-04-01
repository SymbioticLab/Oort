import json, os, ast, pickle
from statistics import mean

folders = os.listdir('trace_2018')

bw_dict = {}

def get_bw_results(measurement):
    for i in measurement:
        if "tcp_speed_results" in i["values"]:
            bw = ast.literal_eval(i["values"]["tcp_speed_results"])
            if i["id"] not in bw_dict:
                bw_dict[i["id"]] = {}
            bw_dict[i["id"]][i["timestamp"]] = bw



for folder in folders:
    path = "trace_2018/" + folder + "/"
    traces = os.listdir(path)
    for trace in traces:
        with open(path + trace, 'r') as f:
            temp = f.read()
            measurement = json.loads(temp)
            get_bw_results(measurement)

print(len(bw_dict))
for key, value in bw_dict.items():
    if len(value) > 1:
        print(len(value))

filename = "trace_result_2018"
outfile = open(filename,'wb')
pickle.dump(bw_dict, outfile, protocol=2)
outfile.close()
