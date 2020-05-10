import pickle, math
import numpy as np
import gurobipy as gp
from gurobipy import *
import time, sys
from queue import PriorityQueue

data_trans_size = 100663 # size of mobilenet, 12 MB
budget = 200

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
def select_by_sorted_num(datas, pref, budget):
    maxTries = 1000
    curTries = 0

    interestChanged = True
    sum_of_cols = None
    listOfInterest = None
    num_rows = len(datas)

    clientsTaken = {}
    data_copy = np.copy(datas)

    preference = {k:pref[k] for k in pref}

    while len(preference.keys()) > 0 and len(clientsTaken) < budget:

        print('curTries: {}, Remains preference: {}, picked clients {}'
                .format(curTries, len(preference.keys()), len(clientsTaken)))

        # recompute the top-k clients
        if interestChanged:
            listOfInterest = list(preference.keys())
            sum_of_cols = sum_interest_columns(np.copy(datas), listOfInterest)

        # start greedy until exceeds budget or interestChanged

        # calculate sum of each client
        top_k_indices = sorted(range(num_rows), reverse=True, key=lambda k:np.sum(sum_of_cols[k]))

        # the rest is also zero, no need to move on
        if sum_of_cols[top_k_indices[0]] == 0:
            break

        for clientId in top_k_indices:

            if sum_of_cols[clientId] == 0:
                break

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

            # 2: remove client from data by setting to zero
            datas[clientId, :] = 0
            clientsTaken[clientId] = tempTakenSamples

            if interestChanged:
                break

        curTries += 1

    # check preference 
    is_success = (len(preference) == 0 and len(clientsTaken) <= budget)

    checkPref = {k:0 for k in pref}

    for client in clientsTaken:
        for cl in clientsTaken[client]:
            assert(clientsTaken[client][cl] <= data_copy[client][cl])
            checkPref[cl] += clientsTaken[client][cl]

    if is_success:
        for cl in sorted(pref.keys()):
            assert(pref[cl] == checkPref[cl])

    print("Picked {} clients".format(len(clientsTaken)))
    return clientsTaken, is_success

def cal_jct(clientsTaken, sys):
    duration = []
    for client in clientsTaken:
        dur = sum([clientsTaken[client][cl] for cl in client])*sys[client][0] + data_trans_size/sys[client][1]
        duration.append(dur)

    return duration


#run_heuristic()

#################### LP ##############################


def lp_solver(datas, systems, budget, preference, data_trans_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget=True):

    num_of_clients = len(datas)
    num_of_class = len(datas[0])

    # Create a new model
    m = gp.Model("client_selection")

    qlist = []
    for i in range(num_of_clients):
        for j in range(num_of_class):
            qlist.append((i, j))

    slowest = m.addVar(vtype=GRB.CONTINUOUS, name="slowest", lb = 0.0)
    quantity = m.addVars(qlist, vtype=GRB.INTEGER, name="quantity", lb = 0) # # of example for each class

    # Fan: sys[i][0] is the inference latency, so we should multiply, ms -> sec for inference latency
    time_list = [((gp.quicksum([quantity[(i, j)] for j in range(num_of_class)])*systems[i][0])/1000. + data_trans_size/systems[i][1]) for i in range(num_of_clients)]

    # The objective is to minimize the slowest
    m.setObjective(slowest, GRB.MINIMIZE)

    # Minimize the slowest
    for t in time_list:
        m.addConstr(slowest >= t, name='slow')

    # Preference Constraint
    for i in range(num_of_class):
        m.addConstr(gp.quicksum([quantity[(client, i)] for client in range(num_of_clients)]) >= preference[i], name='preference_' + str(i))

    # Capacity Constraint
    m.addConstrs((quantity[i] <= datas[i[0]][i[1]] for i in qlist), name='capacity_'+str(i))

    # Budget Constraint
    if request_budget:
        status = m.addVars([i for i in range(num_of_clients)], vtype = GRB.BINARY, name = 'status') # Binary var indicates the selection status
        for i in range(num_of_clients):
            m.addGenConstrIndicator(status[i], False, gp.quicksum([quantity[(i, j)] for j in range(num_of_class)]) ==  0.0)
        m.addConstr(gp.quicksum([status[i] for i in range(num_of_clients)]) <= budget, name = 'budget')


    # Initialize variables if init_values is provided
    if init_values: 
        for k, v in init_values.items():
            quantity[k].Start = v

    # Set a 'time_limit' second time limit
    if time_limit:
        m.Params.timeLimit = time_limit

    m.update()
    if read_flag:
        if os.path.exists('temp.mst'):
            m.read('temp.mst')

    m.optimize()
    print(f'Optimization took {m.Runtime}')
    print(f'Gap between current and optimal is {m.MIPGap}')

    # Process the solution
    result = [[0] * num_of_class for _ in range(num_of_clients)]
    if m.status == GRB.OPTIMAL:
        print('Found optimal solution')
        pointx = m.getAttr('x', quantity)
        for i in qlist:
            if quantity[i].x > 0.0001:
                #print(i, pointx[i])
                result[i[0]][i[1]] = pointx[i]
    else:
        print('No optimal solution')
    
    if write_flag:
        m.write('temp.mst')

    return result

def preprocess(data):
    # Get the global distribution
    distr = [0] * len(data[0])

    for i in data:
        distr = [sum(x) for x in zip(distr, i)]

    outfile = 'global_distr'
    outf = open(outfile, 'wb')
    pickle.dump(global_distr, outf)
    outf.close()

    return None

def run_lp():
    global budget, data_trans_size

    data, systems, distr = load_profiles('openImg_size.txt', 'clientprofile', 'openImg_global_distr')

    num_of_class = 596 #len(data[0])
    data = data[:, :num_of_class]

    sys_prof = [systems[i+1] for i in range(len(data))] # id -> speed, bw
    preference = [math.floor(i * 0.1) for i in distr[:num_of_class]]

    start_time = time.time()
    result = lp_solver(data, sys_prof, budget, preference, data_trans_size)
    finish_time = time.time()

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
            print("Preference not satisfied")

    if flag:
        print("Perfect")


def run_heuristic():
    global budget, data_trans_size

    data, systems, distr = load_profiles('openImg_size.txt', 'clientprofile', 'openImg_global_distr')
    num_of_class = 596 #len(data[0])
    num_of_clients = len(data)

    data = data[:num_of_clients, :num_of_class]

    preference = [math.floor(i * 0.1) for i in distr[:num_of_class]]
    preference_dict = {idx:p for idx, p in enumerate(preference)}
    avg_of_pref_samples = sum(preference)/float(budget)
    sys_prof = [systems[i+1] for i in range(num_of_clients)] # id -> speed, bw
    
    # sort the client by speed
    # est_dur = []
    # for idx in range(num_of_clients):
    #     #est_dur.append(avg_of_pref_samples * sys_prof[idx][0]+data_trans_size/sys_prof[idx][1])
    #     est_dur.append(1./sys_prof[idx][0]*sys_prof[idx][1])

    #top_clients = sorted(range(num_of_clients), key=lambda k:est_dur[k])

    # sort clients by # of samples
    sum_sample_per_client = sum_interest_columns(np.copy(data), list(preference_dict.keys()))
    top_clients = sorted(range(num_of_clients), reverse=True, key=lambda k:np.sum(sum_sample_per_client[k])) #sorted(range(num_of_clients), key=lambda k:est_dur[k])
    
    # random.shuffle(top_clients)

    # decide the cut-off
    cut_off_clients = int(1 * num_of_clients)
    select_clients = None

    start_time = time.time()
    while True:
        tempData = data[top_clients[:cut_off_clients], :]
        clientsTaken, is_success = select_by_sorted_num(tempData, preference_dict, budget)

        if is_success:
            # paraphrase the client IDs given cut_off
            select_clients = {top_clients[k]:clientsTaken[k] for k in clientsTaken.keys()}

            # pad the budget
            if len(select_clients) < budget:
                for client in top_clients:
                    if client not in select_clients:
                        select_clients[client] = {}

                    if len(select_clients) == budget:
                        break
            break
        else:
            cut_off_clients = min(cut_off_clients * 2, num_of_clients)
            print("====Augment the cut_off_clients to {}".format(cut_off_clients))

    print("====Client augmentation took {} sec to pick {} clients".format(time.time() - start_time, len(select_clients)))

    #=================== Stage 2: pass to LP ===================#

    select_client_list = list(select_clients.keys())
    tempdata = data[select_client_list, :]
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
    result = lp_solver(tempdata, tempsys, budget, preference, data_trans_size, init_values=init_values, request_budget=False)
    finish_time = time.time()

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
            print("Preference not satisfied")

    if flag:
        print("Perfect")

run_heuristic()
#run_lp()

