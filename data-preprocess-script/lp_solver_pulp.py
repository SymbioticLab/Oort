from pulp import *
import pickle
import numpy as np 

def lp_solver(datas, systems, budget, cost, preference, bw, data_trans_size):

    num_of_clients = len(datas)
    num_of_class = len(datas[0])
    prob = LpProblem("Client_selection", LpMinimize)
    
    
    qlist = []
    for idx, data in enumerate(datas):
        for i in range(len(datas[0])):
            qlist.append((idx, i))


    quantity = LpVariable.dicts("Quantity",
                                     qlist,
                                     lowBound=0,
                                     cat='Integer')

    status = LpVariable.dicts("selection_status",
                                     [i for i in range(num_of_clients)],
                                     cat='Binary')
    print(type(status))
    slowest = LpVariable("slowest", 0)

    # Objective
    time_list = [((lpSum([quantity[(i, j)] for j in range(num_of_class)])/systems[i]) + data_trans_size/bw[i]) for i in range(num_of_clients)]
    prob += slowest

    # Minimize the slowest
    for t in time_list:
        prob += slowest >= t


    # Preference Constraint
    for i in range(num_of_class):
        prob += lpSum([quantity[(client, i)] for client in range(num_of_clients)]) >= preference[i]

    # Capacity Constraint
    for i in qlist:
        prob += quantity[i] <= datas[i[0]][i[1]]

    # Budget Constraint
    # print([status[i] for i in status])
    # prob += lpSum([status[i] for i in range(num_of_clients)]) <= 2
    # count_list = []
    # for i in range(num_of_clients):
    #     count = 0
    #     for j in range(num_of_class):
            
    #     count_list.append(count)

    # prob += lpSum(count_list) <= 2


    prob.solve()
    print(LpStatus[prob.status])
    for v in prob.variables():
        if v.varValue > 0:
            print(v.name, "=", v.varValue)


datas = [[10, 20, 10, 1], [0, 19, 1, 5], [7, 0, 10, 9], [0, 0, 1, 10]]
system = [14, 10, 17, 10]
bw = [2, 5, 5, 10]
data_trans_size = 5
cost = [1, 1, 1, 1]
budget = 2
preference = [10, 20, 15, 20]
lp_solver(datas, system, budget, cost, preference, bw, data_trans_size)
