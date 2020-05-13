import cplex


def lp_solver(datas, systems, budget, preference, data_trans_size, init_values = None, time_limit = None, read_flag = False, write_flag = False, request_budget=True):


    num_of_clients = len(datas)
    num_of_class = len(datas[0])

    # Create the modeler/solver
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)


    trans_time = [round(100/bw[i], 2) for i in range(num_of_clients)]
    speed = [systems[i]/1000. for i in range(num_of_clients)]

    slowest = list(prob.variables.add(obj = [1.0], lb = [0.0], types = ['C'], names = ['slowest']))



    quantity = [None] * num_of_clients
    for i in range(num_of_clients):
        quantity[i] = list(prob.variables.add(obj = [0.0] * num_of_class,
                                              lb = [0.0] * num_of_class,
                                              ub = [q for q in datas[i]],
                                              types = ['I'] * num_of_class,
                                              names = [f'Client {i} Class{j}' for j in range(num_of_class)]))


    # slowest - time >= 0
    for i in range(num_of_clients):
        ind = slowest + [quantity[i][j] for j in range(num_of_class)]
        #print(ind)
        val = [1.0] + [-speed[i]] * num_of_class
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                    senses = ['G'],
                                    rhs = [trans_time[i]],
                                    names = [f'slow_{i}'])

    for j in range(num_of_class):
        ind = [quantity[i][j] for i in range(num_of_clients)]
        print(ind)
        val = [1.0] * num_of_clients
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                                    senses = ['G'],
                                    rhs = [preference[j]],
                                    names = [f'preference_{i}'])


    # Solve the problem
    prob.solve()

    # And print the solutions
    print("Solution status =", prob.solution.get_status_string())
    print("Optimal value:", prob.solution.get_objective_value())
    tol = prob.parameters.mip.tolerances.integrality.get()
    values = prob.solution.get_values()
    print(values)
    print("preference is " + str(preference))
    for i in range(num_of_clients):
        for j in range(num_of_class):
            if values[quantity[i][j]] > tol:
                print(f"Client {i} Class {j} : {values[quantity[i][j]]}")




#datas = [[10, 20, 10, 2, 2], [10, 19, 10, 1, 3], [20, 15, 17, 20, 5], [20, 30, 17, 10, 7]]
datas = [[10, 20, 10, 1], [0, 19, 0, 0], [0, 0, 1, 0], [0, 0, 1, 10]]
system = [10, 10, 17, 10]
#system = [5 , [5241, 14], [15132, 5], [30123, 9]]
#system = [5, 10, 15, 14]
#datas = [[100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100]]
cost = [1, 2, 3, 5]
budget = 10
data_trans_size = 100
bw = [0.001, 2, 0.001, 0.001]
#preference = [2, 2, 2, 2, 2]
preference = [10, 39, 11, 10]
lp_solver(datas, system, budget, preference, bw, data_trans_size)