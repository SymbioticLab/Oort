import pickle
import numpy as np 
from pulp import *


def lp_solver(datas, systems, budget, cost):

    data_trans_size = 5
    num_of_clients = len(datas)
    prob = LpProblem("Client_selection", LpMinimize)
    
    
    qlist = []
    for idx, data in datas:
        for i in range(len(datas[0])):
            qlist.append((idx, i))


    quantity = LpVariable.dicts("Quantity",
                                     qlist,
                                     lowBound=0,
                                     cat='Integer')

    status = LpVariable.dicts("selection_status",
                                     [i for i in range(num_of_clients)],
                                     cat='Binary')


    # Objective
    prob += max([lpSum()/ + data_trans_size/bw[i] for i in ranage(num_of_clients)])


    # Preference Constraint
    prob += 

    # Capacity Constraint


    # Budget Constraint
    prob +=lpSum([cost[i] * status[i] for i in range(num_of_clients)]) <= budget 


def load_profiles(datafile, sysfile):
    # load user data information
    datas = pickle.load(open(datafile, 'rb'))

    # load user system information
    systems = None #pickle.load(open(sysfile, 'rb'))

    return datas, systems

def sum_interest_columns(datas, cols):
    sum_of_cols = datas[:, cols[0]]

    for col in cols[1:]:
        sum_of_cols += datas[:, col]

    return sum_of_cols

#### Step 1: meet data preference first ####
def select_by_sorted_num(datas, preference, budget):
    maxTries = 10000
    curTries = 0

    interestChanged = True
    sum_of_cols = None
    listOfInterest = None

    clientsTaken = {}

    while curTries < maxTries and len(preference.keys()) > 0:

        # recompute the top-k clients
        if interestChanged:
            listOfInterest = list(preference.keys())
            sum_of_cols = sum_interest_columns(datas, listOfInterest)

        # start greedy until exceeds budget or interestChanged

        # calculate sum of each client
        top_k_indices = sorted(range(len(sum_of_cols)), reverse=True, lambda k:sum(sum_of_cols[k]))

        for clientId in top_k_indices:
            
            # update quantities
            # 1. update preference
            tempTakenSamples = {}
            for cl in listOfInterest:
                takenSamples = max(preference[cl], sum_of_cols[clientId][cl])
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

    return clientsTaken

#### Step 2: greedily optimize the straggler ####
def optimize_straggler():
    pass


def lp_heuristic():
    datas, systems = load_profiles('sortedOpenImg', '')

    # randomly generating preference
    pref = [4000 for i in range(len(datas[0]))]
    print(select_by_sorted_num(datas, pref, 200))


