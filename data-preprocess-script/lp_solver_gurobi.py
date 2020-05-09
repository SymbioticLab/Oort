import pickle, math
import numpy as np
import gurobipy as gp
from gurobipy import *
import time

def load_profiles(datafile, sysfile, distrfile):
    # load user data information
    datas = pickle.load(open(datafile, 'rb'))

    # load user system information
    systems = pickle.load(open(sysfile, 'rb'))

    distr = pickle.load(open(distrfile, 'rb'))

    return datas, systems, distr


#################### Heuristic #######################


def sum_interest_columns(datas, cols):
    sum_of_cols = datas[:, cols[0]]

    for col in cols[1:]:
        sum_of_cols += datas[:, col]

    return sum_of_cols

class pacer():
    def __init__(self, datas, cols, topk):
        self.clients = {}
        self.top_k_clients = {}
        self.datas = datas
        self.cols = cols
        self.min_samples = 0
        self.min_id = -1
        self.topk = topk

    def sum_interest_columns_client(self, clientId):
        temp = self.datas[clientId, self.cols[0]]

        for col in self.cols[1:]:
            temp += self.datas[clientId, col]

        return temp

    def register(self, clientId):
        sum_sample = self.sum_interest_columns_client(clientId)
        self.clients[clientId] = sum_sample

        # if len(self.clientId) == self.topk:
        #     # start to replace the min
        #     for 
        # elif 
        # if self.min_id == -1:
        #     self.min_id = clientId
        #     self.min_samples = sum_samples
        # else:
            

#### Step 1: greedily pool enough clients ####
def augment_clients(datas, sys, data_trans_size, pref):
    # estimate the system speed: communication + (avg size/speed)
    num_of_clients = len(datas)
    sum_samples = 0

    pacer = pacer(datas, pref)

    for cl in pref:
        sum_samples += pref[cl]

    est_dur = []
    for idx in range(num_of_clients):
        est_dur.append(data_trans_size/sys[idx][1] + sum_samples * sys[idx][0])


    # sort clients by est_dur from the fastest
    top_clients = sorted(range(num_of_clients), key=lambda k:est_dur[k])

    # put clients one by one until we can get a solution for data preference
    # solution: sum of samples of top-K(budget) clients exceeds the preference

    # fix clients first? then LP opt JCT?
    # for client in top_clients:


#### Step 2: meet data preference first ####
def select_by_sorted_num(datas, preference, budget):
    maxTries = 10000
    curTries = 0

    interestChanged = True
    sum_of_cols = None
    listOfInterest = None

    clientsTaken = {}

    pref = {k:preference[k] for k in preference}

    while curTries < maxTries and len(preference.keys()) > 0:

        print('curTries: {}, Remains preference: {}, picked clients {}, \n Preference: {}'
                .format(curTries, len(preference.keys()), len(clientsTaken), preference))

        # recompute the top-k clients
        if interestChanged:
            listOfInterest = list(preference.keys())
            sum_of_cols = sum_interest_columns(datas, listOfInterest)

        # start greedy until exceeds budget or interestChanged

        # calculate sum of each client
        top_k_indices = sorted(range(len(sum_of_cols)), reverse=True, key=lambda k:np.sum(sum_of_cols[k]))

        for clientId in top_k_indices:
            
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
    checkPref = {k:0 for k in pref}

    for client in clientsTaken:
        for cl in clientsTaken[client]:
            checkPref[cl] += clientsTaken[client][cl]

    for cl in pref:
        assert(pref[cl] == checkPref[cl])
        print(cl, checkPref[cl])

    print("Picked {} clients".format(len(clientsTaken)))
    return clientsTaken

def cal_jct(clientsTaken, sys):
    duration = []
    for client in clientsTaken:
        dur = sum([clientsTaken[client][cl] for cl in client])*sys[client][0] + data_trans_size/sys[client][1]
        duration.append(dur)

    return duration

#### Step 3: greedily optimize the straggler ####
def optimize_straggler():
    pass


def run_heuristic():
    data, systems, distr = load_profiles('openImg_size.txt', 'clientprofile', 'openImg_global_distr')

    num_of_class = 100
    preference = {i:math.floor(distr[i] * 0.1) for i in range(num_of_class)}

    select_by_sorted_num(data, preference, 300)

#run_heuristic()

#################### LP ##############################
def lp_solver(datas, systems, budget, preference, data_trans_size):

    num_of_clients = len(datas)
    num_of_class = len(datas[0])
    #num_of_clients = 596
    num_of_class = 596

    # Create a new model
    m = gp.Model("client_selection")

    qlist = []
    for i in range(num_of_clients):
        for j in range(num_of_class):
            qlist.append((i, j))

    slowest = m.addVar(vtype=GRB.CONTINUOUS, name="slowest", lb = 0.0)
    quantity = m.addVars(qlist, vtype=GRB.INTEGER, name="quantity", lb = 0) # # of example for each class
    status = m.addVars([i for i in range(num_of_clients)], vtype = GRB.BINARY, name = 'status') # Binary var indicates the selection status

    # Fan: sys[i][0] is the inference latency, so we should multiply
    time_list = [((gp.quicksum([quantity[(i, j)] for j in range(num_of_class)])*systems[i][0]) + data_trans_size/systems[i][1]) for i in range(num_of_clients)]

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
    for i in range(num_of_clients):
        m.addGenConstrIndicator(status[i], False, gp.quicksum([quantity[(i, j)] for j in range(num_of_class)]) ==  0.0)

    m.addConstr(gp.quicksum([status[i] for i in range(num_of_clients)]) <= budget, name = 'budget')

    start_time = time.time()
    m.optimize()

    print('optimize is {}'.format(time.time() - start_time))
    result = [[0] * num_of_class for _ in range(num_of_clients)]
    # Print Solution
    if m.status == GRB.OPTIMAL:
        print('Found solution')
        pointx = m.getAttr('x', quantity)
        for i in qlist:
            if quantity[i].x > 0.0001:
                #print(i, pointx[i])
                result[i[0]][i[1]] = pointx[i]
    else:
        print('No solution')

    finish_time = time.time()

    print('Duration is {}'.format(finish_time - start_time))
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
    data, systems, distr = load_profiles('openImg_size.txt', 'clientprofile', 'openImg_global_distr')

    sys_prof = [systems[i+1] for i in range(len(data))] # id -> speed, bw
    preference = [math.floor(i * 0.1) for i in distr]
    budget = 400
    data_trans_size = 100663 # size of mobilenet, 12 MB

    start_time = time.time()
    result = lp_solver(data, sys_prof, budget, preference, data_trans_size)
    finish_time = time.time()

    print("LP solver takes {} sec".format(finish_time - start_time))

    temp = [0] * len(data[0])
    count = 0
    for i in result:
        if sum(i) >= 1:
            count += 1
        temp = [sum(x) for x in zip(temp, i)]

    print(count)
    flag = True
    for i, j in zip(temp, preference):
        if i < j:
            flag = False
            print("Preference not satisfied")

    if flag:
        print("Perfect")

run_lp()
