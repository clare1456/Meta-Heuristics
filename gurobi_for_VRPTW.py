# applying gurobipy to solve VRPTW

import gurobipy as gp
from gurobipy import GRB
from read_data import read_data
import numpy as np
import matplotlib.pyplot as plt
from time import time

def gurobi_VRPTW(problem, show = False):
    # read data
    vehicle_num = problem.vehicle_num
    capacity = problem.vehicle_capacity
    customers = problem.customers # depot index 0, including x, y, demand, ready_time, due_time, service_time
    speed = 1 # set speed subjectively
    M = 1e7 # set a big number
    # seperate data
    positions = customers[:, :2]
    demands = customers[:, 2]
    ready_times = customers[:, 3]
    due_times = customers[:, 4]
    service_time = customers[:, 5]

    # building model
    MODEL = gp.Model('VRPTW')

    p_num = len(customers)
    points = list(range(p_num))
    A = [(i, j) for i in points for j in points]
    D = {(i, j): np.linalg.norm(positions[i]-positions[j]) for i, j in A}

    ## add variates
    x = MODEL.addVars(A, vtype=GRB.BINARY)
    s = MODEL.addVars(points, vtype=GRB.CONTINUOUS)
    c = MODEL.addVars(points, vtype=GRB.CONTINUOUS)
    ## set objective
    MODEL.modelSense = GRB.MINIMIZE
    MODEL.setObjective(gp.quicksum(x[i, j] * D[i, j] for i, j in A))
    ## set constraints
    ### 1. flow balance
    MODEL.addConstrs(gp.quicksum(x[i, j] for j in points if j!=i)==1 for i in points[1:]) # depot not included
    MODEL.addConstrs(gp.quicksum(x[i, j] for i in points if i!=j)==1 for j in points[1:]) # depot not included
    ### 2. avoid subring / self-loop
    MODEL.addConstrs(s[i] + D[i, j] / speed + service_time[i] - M * (1 - x[i, j]) <= s[j] for i, j in A if j!=0)
    ### 3. time constraints
    MODEL.addConstrs(s[i] >= ready_times[i] for i in points)
    MODEL.addConstrs(s[i] <= due_times[i] for i in points)
    ### 4. capacity constraints
    MODEL.addConstrs(c[i] - demands[j] + M * (1 - x[i, j])>= c[j] for i, j in A if j!=0)
    MODEL.addConstrs(c[i] <= capacity for i in points)
    MODEL.addConstrs(c[i] >= 0 for i in points)
    ### 5. vehicle number constraint
    MODEL.addConstr(gp.quicksum(x[0, j] for j in points) <= vehicle_num)

    # optimize the model
    MODEL.optimize()

    # get the routes
    routes = []
    for j in range(1, p_num):
        if round(x[0, j].X) == 1:
            route = [0]
            route.append(j)
            i = j
            while j != 0:
                for j in range(p_num):
                    if round(x[i, j].X) == 1:
                        route.append(j)
                        i = j
                        break
            routes.append(route)
 
    return routes, MODEL.ObjVal

def show_routes(positions, routes):
    plt.figure()
    plt.scatter(positions[1:, 0], positions[1:, 1])
    plt.scatter(positions[0:1, 0], positions[0:1, 1], s = 150, c = 'r', marker='*')
    for route in routes:
        plt.plot(positions[route, 0], positions[route, 1], c='r')
    plt.show()

if __name__ == "__main__":
    file_name = "solomon_100/C101.txt"
    # 101\102-1s, 101_25-0.02s, 103-200s
    problem = read_data(file_name)
    time1 = time()
    routes, obj = gurobi_VRPTW(problem, show=True)
    time2 = time()
    show_routes(problem.customers[:, :2], routes)
    print("optimal obj: {}\ntime consumption: {}".format(obj, time2-time1))
