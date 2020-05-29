import pickle, math
import numpy as np
import gurobipy as gp
from gurobipy import *
import time, sys, gc
from queue import PriorityQueue
from numpy import *
from lp_solver import *

sys.stdout.flush()

data_trans_size = 100663 # size of mobilenet, 12 MB
budget = 100
gap_bar = 0.25 # terminate the lp
num_of_class_interest = 596 #596
greedy_only = False

def load_profiles(datafile, sysfile, distrfile):
    # load user data information
    datas = pickle.load(open(datafile, 'rb'))

    # load user system information
    systems = pickle.load(open(sysfile, 'rb'))

    # Load global data distribution
    distr = pickle.load(open(distrfile, 'rb'))

    return datas, systems, distr


#################### Heuristic #######################

class Pacer():
    def __init__(self, datas, pref, topk):
        self.clients = PriorityQueue()
        self.top_k_clients = {}
        self.datas = datas
        self.pref = pref

        self.topk = topk
        self.cols = list(pref.keys())

        self.pref_window = {k: self.pref[k] for k in self.pref}
        self.isfeasible = False

        self.checkpoint = []
        self.temp_init = [0 for i in range(len(pref))]

    def sum_interest_columns_client(self, clientId):
        temp = self.datas[clientId, self.cols[0]]

        for col in self.cols[1:]:
            temp += self.datas[clientId, col]

        return temp

    def replace_client(self, prev, cur):
        # evict prev from self.pref_window
        for cl in self.cols:
            self.pref_window[cl] += self.top_k_clients[prev][cl]
        del self.top_k_clients[prev]

        # push the current id
        self.push_client_window(cur)

    def push_client_window(self, clientId):
        temp_sum = 0

        self.top_k_clients[clientId] = self.temp_init

        for cl in self.cols:
            samplesTaken = min(self.datas[clientId, cl], self.pref_window[cl])
            self.top_k_clients[clientId][cl] = samplesTaken
            self.pref_window[cl] -= samplesTaken

            temp_sum += self.pref_window[cl]

        if temp_sum == 0:
            self.isfeasible = True
            self.checkpoint.append(self.top_k_clients)

    def register(self, clientId):
        sum_sample = self.sum_interest_columns_client(clientId)

        if self.clients.qsize() <= self.topk:
            self.push_client_window(clientId)
            self.clients.put((sum_sample, clientId))
        else:
            # retain a window
            currentMin = self.clients.get()
            if sum_sample > currentMin[0]:
                # replace the window
                self.replace_client(currentMin[1], clientId)
                self.clients.put((sum_sample, clientId))
            else:
                self.clients.put(currentMin)

    def find_feasible(self):
        return self.isfeasible

    def feasible_sols(self):
        return self.checkpoint


#### Step 1: greedily pool enough clients ####
def augment_clients(datas, sys, data_trans_size, pref, budget):
    # estimate the system speed: communication + (avg size/speed)
    num_of_clients = len(datas)
    sum_samples = 0

    for cl in pref:
        sum_samples += pref[cl]

    est_dur = []
    for idx in range(num_of_clients):
        est_dur.append(data_trans_size/sys[idx][1] + sum_samples * sys[idx][0])


    # sort clients by est_dur from the fastest
    top_clients = sorted(range(num_of_clients), key=lambda k:est_dur[k])

    # put clients one by one until we can get a solution for data preference
    # solution: sum of samples of top-K(budget) clients exceeds the preference

    pacer = Pacer(datas, pref, budget)
    cut_off = len(top_clients)

    # fix clients first? then LP opt JCT?
    for idx, client in enumerate(top_clients):
        # push the client one by one util find a feasible solution
        pacer.register(client)

        if pacer.find_feasible():
            cut_off = idx

        if idx > 2 * cut_off:
            break

    # load all feasible solutions
    feasible_sols = pacer.feasible_sols()

    return feasible_sols


def sum_interest_columns(datas, cols):
    sum_of_cols = datas[:, cols[0]]

    for col in cols[1:]:
        sum_of_cols = sum_of_cols + datas[:, col]

    return sum_of_cols

#### Step 2: meet data preference first ####
def select_by_sorted_num(raw_datas, pref, budget):
    maxTries = 1000
    curTries = 0

    interestChanged = True
    sum_of_cols = None
    listOfInterest = None
    num_rows = len(raw_datas)

    clientsTaken = {}
    datas = np.copy(raw_datas)

    preference = {k:pref[k] for k in pref}

    while len(preference.keys()) > 0 and len(clientsTaken) < budget:

        print('curTries: {}, Remains preference: {}, picked clients {}'
                .format(curTries, len(preference.keys()), len(clientsTaken)))

        # recompute the top-k clients
        if interestChanged:
            listOfInterest = list(preference.keys())
            sum_of_cols = datas[:, listOfInterest].sum(axis=1)

        # calculate sum of each client
        # filter the one w/ zero samples
        feasible_clients = np.where(sum_of_cols > 0)[0]

        if len(feasible_clients) == 0:
            break

        top_k_indices = sorted(feasible_clients, reverse=True, key=lambda k:sum_of_cols[k])

        for idx, clientId in enumerate(top_k_indices):
            # update quantities
            # 1. update preference
            tempTakenSamples = {}

            for cl in listOfInterest:
                takenSamples = min(preference[cl], datas[clientId][cl])
                preference[cl] -= takenSamples

                if preference[cl] == 0:
                    del preference[cl]
                    interestChanged = True

                tempTakenSamples[cl] = takenSamples

            datas[clientId, :] = 0
            clientsTaken[clientId] = tempTakenSamples

            if interestChanged:
                break

        curTries += 1

    # check preference
    is_success = (len(preference) == 0 and len(clientsTaken) <= budget)

    print("Picked {} clients".format(len(clientsTaken)))
    return clientsTaken, is_success

def cal_jct(clientsTaken, sys):
    duration = []
    for client in clientsTaken:
        dur = sum([clientsTaken[client][cl] for cl in client])*sys[client][0] + data_trans_size/sys[client][1]
        duration.append(dur)

    return duration


def preprocess(data_file):
    # Get the global distribution
    #distr = [0] * len(data[0])

    with open(data_file, 'rb') as fin:
        data = pickle.load(fin)

    # print(data[23][:100])
    # data = np.array(data)
    # print(data[23][:100])
    print(len(data[0]))
    distr = data.sum(axis=0).tolist()
    print(len(distr))

    outfile = 'so_global_distr'
    outf = open(outfile, 'wb')
    pickle.dump(distr, outf)
    outf.close()

    return None

#preprocess('so_data_distr')

def run_lp(requirement, data, systems, distr):
    global budget, data_trans_size

    num_of_class = num_of_class_interest #len(data[0])
    num_of_clients = len(data)
    distr = distr[:num_of_class]
    sum_distr = sum(distr)

    data = data[:num_of_clients, :num_of_class]

    preference = [math.floor((i/float(sum_distr)) * requirement) for i in distr]

    sys_prof = [systems[i+1] for i in range(len(data))] # id -> speed, bw
    #preference = [math.floor(i * 0.4) for i in distr[:num_of_class]]

    start_time = time.time()
    result, sol, lp_duration = lp_solver(data, sys_prof, budget, preference, data_trans_size, gap=gap_bar)#, request_budget=False)
    finish_time = time.time()
    lp_solver_duration = finish_time - start_time

    print(f"LP solver took {finish_time - start_time} sec")

    temp = [0] * len(data[0])
    count = 0
    for i in result:
        if sum(i) >= 1:
            count += 1
        temp = [sum(x) for x in zip(temp, i)]

    print(f'Selected {count} clients')
    flag = True
    for i, j in zip(temp, preference):
        if i < j:
            flag = False
            #print("Preference not satisfied")

    flag = flag & (sol.status == GRB.OPTIMAL)
    if flag:
        print("Perfect")
        return [lp_duration, sol.objVal, lp_solver_duration, count]
    else:
        return [lp_duration, -1, lp_solver_duration, -1]

def greedy_result(select_clients, sys_prof):
    # get the sum of samples for each client
    task_duration = []

    for clientId in select_clients:
        class_dict = select_clients[clientId]
        samples_taken = sum([class_dict[cl] for cl in class_dict])
        task_duration.append(samples_taken*sys_prof[clientId][0]/1000. + data_trans_size/float(sys_prof[clientId][1]))

    return max(task_duration)

def run_heuristic(requirement, _raw_data, systems, distr):
    global budget, data_trans_size, num_of_class_interest, greedy_only


    #data, systems, distr = load_profiles('openImg_size.txt', 'clientprofile', 'openImg_global_distr')
    # data, systems, distr = load_profiles('so_data_distr', 'clientprofile', 'so_global_distr')
    data = np.copy(_raw_data)

    num_of_class = num_of_class_interest #596 #len(data[0])
    num_of_clients = len(data)
    distr = distr[:num_of_class]
    sum_distr = sum(distr)

    print(num_of_clients, num_of_class)

    data = data[:num_of_clients, :num_of_class]

    raw_data = np.copy(data)


    preference = [math.floor((i/float(sum_distr)) * requirement) for i in distr]
    preference_dict = {idx:p for idx, p in enumerate(preference)}
    avg_of_pref_samples = sum(preference)/float(budget)
    sys_prof = [systems[i+1] for i in range(num_of_clients)] # id -> speed, bw

    # cap the data by preference
    # pref_matrix = tile(array(preference), (num_of_clients, 1))
    # data = np.minimum(data, pref_matrix)

    # sort clients by # of samples
    sum_sample_per_client = data[:, list(preference_dict.keys())].sum(axis=1) #sum_interest_columns(np.copy(data), list(preference_dict.keys()))
    top_clients = sorted(range(num_of_clients), reverse=True, key=lambda k:np.sum(sum_sample_per_client[k]))

    # random.shuffle(top_clients)

    # decide the cut-off
    ratio_of_rep = max([preference[i]/float(distr[i]) * 5.0 for i in range(len(preference)) if distr[i] > 0])
    cut_off_clients = min(int(ratio_of_rep * num_of_clients + 1), num_of_clients)

    print(f"cut_off_clients is {cut_off_clients}")
    select_clients = None

    cut_off_required = 200 #max(budget * 0.1, 100)

    start_time = time.time()
    while True:
        tempData = data[top_clients[:cut_off_clients], :]
        clientsTaken, is_success = select_by_sorted_num(tempData, preference_dict, budget)

        if is_success:
            # paraphrase the client IDs given cut_off
            select_clients = {top_clients[k]:clientsTaken[k] for k in clientsTaken.keys()}

            # pad the budget
            if len(select_clients) < cut_off_required and not greedy_only:
                for client in top_clients:
                    if client not in select_clients:
                        select_clients[client] = {}

                    if len(select_clients) == cut_off_required:
                        break
            break
        else:
            print(f"====Fail to augment with cut_off_clients {cut_off_clients}")
            if cut_off_clients == num_of_clients:
                return [time.time() -  start_time, 0, -1]
            cut_off_clients = min(cut_off_clients * 2, num_of_clients)
            print("====Augment the cut_off_clients to {}".format(cut_off_clients))

    augTime = time.time() - start_time
    print("====Client augmentation took {} sec to pick {} clients".format(augTime, len(select_clients)))

    #=================== Stage 2: pass to LP ===================#

    select_client_list = list(select_clients.keys())
    tempdata = raw_data[select_client_list, :]
    tempsys = [sys_prof[i] for i in select_client_list]

    # load initial value
    init_values = {}

    for clientId in range(len(tempdata)):
        for cl in range(len(tempdata[0])):
            init_values[(clientId, cl)] = 0

    for idx, key in enumerate(select_client_list):
        for cl in select_clients[key]:
            init_values[(idx, cl)] = select_clients[key][cl]

    start_time = time.time()

    if not greedy_only:
        result, sol, lp_duration = lp_solver(tempdata, tempsys, budget, preference, data_trans_size, init_values=init_values, request_budget=False, gap=gap_bar)
    else:
        result = greedy_result(select_clients, sys_prof)
        return [augTime, 0., result, 0., len(select_clients)]

    finish_time = time.time()
    lp_solver_duration = finish_time - start_time

    print(f"LP solver took {finish_time - start_time:.2f} sec")

    temp = [0] * len(data[0])
    count = 0
    for i in result:
        if sum(i) >= 1:
            count += 1
        temp = [sum(x) for x in zip(temp, i)]

    print(f'Selected {count} clients')
    flag = True
    for i, j in zip(temp, preference):
        if i < j:
            flag = False
            print("Preference not satisfied")

    assert(flag == (sol.status == GRB.OPTIMAL))
    if flag:
        print("Perfect")

    print(f'\# of class {num_of_class}, \# of clients {num_of_clients}')

    if flag:
        return [augTime, lp_duration, sol.objVal, lp_solver_duration, count]

    return [augTime, lp_duration, sol.objVal, lp_solver_duration, -1]

#temp_requirements = [1000, 2000, 4000, 5000, 6000, 7000, 8000] + [10000*i for i in range(1, 10)] + [100000*i for i in range(1, 10)] #, 20000, 40000, 60000, 80000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 700000, 800000, 900000, 1000000, 1200000]
temp_requirements = [1000 * i for i in range(1, 11)] + [10000 * i for i in range(2, 11)] + [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 700000, 800000, 900000, 1000000, 1200000]

#temp_requirements = [10000*i for i in range(1, 10)] + [100000*i for i in range(1, 10)]
requirements = [x for x in temp_requirements if x > num_of_class_interest]# * 5]
#requirements = [20000]

is_heuristic = True #True

budgets = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

gc.disable()
#data, systems, distr = load_profiles('/mnt/so/so_data_distr', 'clientprofile', 'so_global_distr')
data, systems, distr = load_profiles('openImg_size.txt', 'clientprofile', 'openImg_global_distr')

gc.enable()


# budget = 500

# output_file = 'class_heuristic_result' if is_heuristic else 'optimal_result'
# with open(output_file, 'w') as fout:
#     classes = [i*10**exp for exp in range(1, 5) for i in range(1, 9)]
#     for i in classes:
#         num_of_class_interest = i

#         print(f"====Start to run {i} class")
#         req = sum(distr[:i]) * 0.01
#         sol = run_heuristic(req, data, systems, distr) if is_heuristic else run_lp(req, data, systems, distr)

#         fout.writelines(f'data_trans_size: {data_trans_size}, class: {i}, requirement: {req}, end-to-end: {sum(sol[:-2]):.3f}, details: {sol} \n')

#         print(f'====data_trans_size: {data_trans_size}, class: {i}, requirement: {req}, end-to-end: {sum(sol[:-2]):.3f}s, details: {sol}\n')
#         if sol[-1] == -1:
#             print(f"====Terminate with {i}")
#             break

#         gc.collect()

greedy_only = False

for item in budgets:
    budget = item
    output_file = str(budget)+'_heuristic_result' if is_heuristic else 'optimal_result'

    with open(output_file, 'w') as fout:
        for i in requirements:
            print(f"====Start to run {i} requirements")
            sol = run_heuristic(i, data, systems, distr) if is_heuristic else run_lp(i, data, systems, distr)

            fout.writelines(f'data_trans_size: {data_trans_size}, budget: {budget}, requirement: {i}, end-to-end: {sum(sol[:-2]):.3f}, details: {sol} \n')

            print(f'====data_trans_size: {data_trans_size}, budget: {budget}, requirement: {i}, end-to-end: {sum(sol[:-2]):.3f}s, details: {sol}\n')
            if sol[-1] == -1:
                print(f"====Terminate with {i}")
                break

            gc.collect()
