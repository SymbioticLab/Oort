import pickle, math
import numpy as np
import gurobipy as gp
from gurobipy import *

def load_profiles(datafile, sysfile, distrfile):
    # load user data information
    datas = pickle.load(open(datafile, 'rb'))

    # load user system information
    systems = pickle.load(open(sysfile, 'rb'))

    distr = pickle.load(open(distrfile, 'rb'))

    return datas, systems, distr

def lp_solver(datas, systems, budget, preference, data_trans_size):

    #num_of_clients = len(datas)
    #num_of_class = len(datas[0])
    num_of_clients = 596
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

    time_list = [((gp.quicksum([quantity[(i, j)] for j in range(num_of_class)])/systems[i][0]) + data_trans_size/systems[i][1]) for i in range(num_of_clients)]

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

    m.optimize()

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
    budget = 1000
    print(preference[:596])
    data_trans_size = 10000


    result = lp_solver(data, sys_prof, budget, preference[:596], data_trans_size)
    temp = [0] * 596
    count = 0
    for i in result:
        if sum(i) >= 1:
            count += 1
        temp = [sum(x) for x in zip(temp, i)]

    print(count)
    flag = True
    for i, j in zip(temp, preference[:596]):
        if i < j:
            flag = False
            print("Preference not satisfied")

    if flag:
        print("Perfect")


run_lp()
