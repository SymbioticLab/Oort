import pickle, math
import numpy as np
import gurobipy as gp
import cplex
from gurobipy import *
import time, sys


def lp_solver(datas, systems, budget, preference, data_trans_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget = True, solver = 'gurobi'):
    if solver == 'gurobi':
        return lp_gurobi(datas, systems, budget, preference, data_trans_size, init_values, time_limit, read_flag, write_flag, request_budget)
    else:
        return lp_cplex(datas, systems, budget, preference, data_trans_size, init_values, time_limit, read_flag, write_flag, request_budget)



def lp_gurobi(datas, systems, budget, preference, data_trans_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget=True):

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


def lp_cplex(datas, systems, budget, preference, data_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget=True):


    num_of_clients = len(datas)
    num_of_class = len(datas[0])

    # Create the modeler/solver
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Time to transmit the data
    trans_time = [round(data_size/systems[i][1], 2) for i in range(num_of_clients)]
    #trans_time = [10 for i in range(num_of_clients)]

    #print(trans_time)
    # System speeds
    speed = [systems[i][0]/1000. for i in range(num_of_clients)]



    slowest = list(prob.variables.add(obj = [1.0], lb = [0.0], types = ['C'], names = ['slowest']))

    quantity = [None] * num_of_clients
    for i in range(num_of_clients):
        quantity[i] = list(prob.variables.add(obj = [0.0] * num_of_class,
                                              lb = [0.0] * num_of_class,
                                              ub = [q for q in datas[i]],
                                              types = ['I'] * num_of_class,
                                              names = [f'Client {i} Class{j}' for j in range(num_of_class)]))


    # Minimize the slowest
    for i in range(num_of_clients):
        ind = slowest + [quantity[i][j] for j in range(num_of_class)]
        #print(ind)
        val = [1.0] + [-speed[i]] * num_of_class
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                    senses = ['G'],
                                    rhs = [trans_time[i]],
                                    names = [f'slow_{i}'])

    # Preference Constraint
    for j in range(num_of_class):
        ind = [quantity[i][j] for i in range(num_of_clients)]
        val = [1.0] * num_of_clients
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                    senses = ['G'],
                                    rhs = [preference[j]],
                                    names = [f'preference_{i}'])

    # Budget Constraint
    if request_budget:
        status = list(prob.variables.add(obj = [0.0] * num_of_clients,
                                         types = ['B'] * num_of_clients,
                                         names = [f'status {i}' for i in range(num_of_clients)]))
        for i in range(num_of_clients):
            ind = [quantity[i][j] for j in range(num_of_class)]
            val = [1.0] * num_of_class
            prob.indicator_constraints.add(indvar=status[i],
                                           complemented=1,
                                           rhs=0.0,
                                           sense="L",
                                           lin_expr=cplex.SparsePair(ind=ind, val=val),
                                           name=f"ind{i}")
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=status, val=[1.0] * num_of_clients)],
                                    senses = ['L'],
                                    rhs = [budget],
                                    names = ['budget'])




    # Solve the problem
    prob.solve()

    # And print the solutions
    print("Solution status =", prob.solution.get_status_string())
    print("Optimal value:", prob.solution.get_objective_value())
    tol = prob.parameters.mip.tolerances.integrality.get()
    values = prob.solution.get_values()
    # print(values)
    # print("preference is " + str(preference))
    # print("System is " + str(systems))
    result = [[0] * num_of_class for _ in range(num_of_clients)]
    for i in range(num_of_clients):
        for j in range(num_of_class):
            if values[quantity[i][j]] > tol:
                result[i][j] = values[quantity[i][j]]
                # print(f"Client {i} Class {j} : {values[quantity[i][j]]}")
    
    return result
