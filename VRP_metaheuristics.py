# Climbing Algorithm solving VRPTW
# author: Charles Lee
# date: 2022.09.24

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from VRP_heuristics import *
from read_data import read_data

# climbing algorithm
class Heuristic():
    def __init__(self, problem, iter_num):
        """ 
        read data and preprocess 
        """
        self.read_data(problem)
        self.iter_num = iter_num
        
    def read_data(self, problem):
        self.problem = problem
        self.vehicle_num = problem.vehicle_num
        self.capacity = problem.vehicle_capacity
        customers = problem.customers # depot index 0, including x, y, demand, ready_time, due_time, service_time
        self.positions = customers[:, :2]
        self.demands = customers[:, 2]
        self.ready_times = customers[:, 3]
        self.due_times = customers[:, 4]
        self.service_time = customers[:, 5]

        self.speed = 1
        self.p_num = len(self.positions)
        self.dist_m = self.cal_distance_matrix(self.positions)
    
        self.choose_neighbour_strategy = "random"
        
        self.process = [] # record data while alg running

    def cal_distance_matrix(self, positions):
        '''
        calculate to get distance matrix
        '''
        dist_m = np.zeros((self.p_num, self.p_num))
        for i in range(self.p_num):
            for j in range(self.p_num):
                dist_m[i, j] = sum((positions[i] - positions[j])**2)**(1/2)
        return dist_m

    def solution_init(self, strategy="heuristic"):
        '''
        generate initial solution (routes), applying VRP_heuristics
        '''
        if strategy == "heuristic":
            alg = Solomon_Insertion(self.problem)
            routes = alg.run()
            # routes = nearest_neighbour(problem)
            solution = []
            for ri, route in enumerate(routes):
                solution.append(0)
                solution += route[1:-1]
            solution.append(0) # add the end 0
        elif strategy == "random":
            solution = list(range(1, self.p_num)) + [0]*(7-1) # 7 vehicles indefault
            solution.shuffle()
            solution = [0] + solution + [0]
        return solution

    def transfer(self, solution):
        """
        transfer solution to routes
        """
        routes = []
        for i, p in enumerate(solution[:-1]): # pass the end 0
            if p == 0:
                if i > 0:
                    routes[-1].append(0) # add end 0
                routes.append([0]) # add start 0
            else:
                routes[-1].append(p)
        else:
            routes[-1].append(0) # add final 0
        return routes

    def cal_objective(self, solution):
        '''
        calculate objective of solution (including consideration of soft/hard constraint)
        '''
        routes = self.transfer(solution)
        obj = 0
        for i in range(len(routes)):
            route = routes[i]
            load = 0
            cur_time = 0
            for j in range(1, len(route)):
                # consideration of distance
                ri = route[j-1]
                rj = route[j]
                distance = self.dist_m[ri, rj]
                obj += distance

                # consideration of capacity
                load += self.demands[ri]
                if load > self.capacity: # break the capacity constraint
                    obj += 1000

                # consideration of time window
                cur_time += self.service_time[ri] + self.dist_m[ri, rj]
                cur_time = max(cur_time, self.ready_times[rj]) # if arrived early, wait until ready
                if cur_time > self.due_times[rj]: # break the TW constraint
                    obj += 0
        return obj
    
    def get_neighbours(self, solution, operator=Relocate()):
        neighbours =  operator.run(solution)
        return neighbours

    def choose_neighbour(self, neighbours):
        # randomly choose neighbour
        if self.choose_neighbour_strategy == "random":
            chosen_ni = np.random.randint(len(neighbours))
        # choose the first neighbour
        if self.choose_neighbour_strategy == "first":
            chosen_ni = 0
        # choose the best neighour
        elif self.choose_neighbour_strategy == "best":
            best_obj = np.inf
            for ni, neighbour in enumerate(neighbours):
                obj = self.cal_objective(neighbour) 
                if obj < best_obj:
                    best_obj = obj
                    best_ni = ni
            chosen_ni = best_ni
        
        return chosen_ni

    def draw(self, routes):
        positions = self.positions
        plt.scatter(positions[:, 0], positions[:, 1])
        for route in routes:
            # add depot 0
            x = list(positions[route, 0])
            x.append(positions[route[0], 0])
            y = list(positions[route, 1])
            y.append(positions[route[0], 1])
            plt.plot(x, y)
        plt.show()

    def show_process(self):
        y = self.process
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.show()
   
    def run(self):
        best_solution = self.solution_init() # solution in form of routes
        best_obj = self.cal_objective(best_solution)
        neighbours = self.get_neighbours(best_solution)
        for step in trange(self.iter_num):
            ni = self.choose_neighbour(neighbours)
            cur_solution = neighbours[ni]
            cur_obj = self.cal_objective(cur_solution)
            # obj: minimize the total distance 
            if cur_obj < best_obj: 
                best_solution = cur_solution
                best_obj = cur_obj
                neighbours = self.get_neighbours(best_solution)
            else:
                neighbours.pop(ni)
                if len(neighbours) == 0:
                    print('local optimal, break out, iterated {} times'.format(step))
                    break

            self.process.append(best_obj)
        self.best_solution = best_solution
        self.best_obj = best_obj
        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes, self.best_obj

# tabu search algorithm
class Tabu_Search(Heuristic):
    def __init__(self, problem, iter_num):
        """ 
        read data and preprocess 
        """
        self.iter_num = iter_num
        self.read_data(problem) 

        # set paraments of tabu search
        self.tabu = []
        self.tabu_length = self.p_num # set tabu length as points number
    
    def tabu_neighbours(self, neighbours):
        ni = 0
        while ni < len(neighbours):
            neighbour = neighbours[ni]
            if neighbour in self.tabu:
                neighbours.pop(ni)
            else:
                ni += 1
        return neighbours

    def run(self):
        local_solution = self.solution_init()
        local_obj = self.cal_objective(local_solution)
        self.best_solution = local_solution.copy()
        self.best_obj = local_obj
        neighbours = self.get_neighbours(local_solution)
        neighbours = self.tabu_neighbours(neighbours)
        for step in trange(self.iter_num):
            ni = self.choose_neighbour(neighbours)
            cur_solution = neighbours[ni]
            cur_obj = self.cal_objective(cur_solution)
            # obj: minimize the total distance 
            if cur_obj < local_obj: 
                local_solution = cur_solution
                local_obj = cur_obj
                neighbours = self.get_neighbours(local_solution)
                neighbours = self.tabu_neighbours(neighbours)
            else:
                if len(neighbours) <= 1:
                    self.tabu.append(local_solution)
                    if len(self.tabu) > self.tabu_length:
                        self.tabu.pop(0)
                    local_solution = neighbours[ni]
                    local_obj = cur_obj
                    neighbours = self.get_neighbours(local_solution)
                    neighbours = self.tabu_neighbours(neighbours)
                else:
                    neighbours.pop(ni)
            if local_obj < self.best_obj:
                self.best_obj = local_obj
                self.best_solution = local_solution

            self.process.append([local_obj, self.best_obj])

        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes, self.best_obj

# simulated annealing algorithm 
class Simulated_Annealing(Heuristic):
    def __init__(self, problem, iter_num):
        """ 
        read data and preprocess 
        """
        self.read_data(problem)
        self.iter_num = iter_num

        # set paraments of SA
        self.max_temp = 10
        self.min_temp = 0
        self.a = 0.997 
        self.a_steps = 40
        
    def SA_accept(self, detaC, temperature):
        return math.exp(-detaC / temperature)

    def temperature_update(self, temperature, step):
        if step % self.a_steps == 0: # update temperature by static steps
            temperature *= self.a
        temperature = max(self.min_temp, temperature)
        return temperature

    def run(self):
        local_solution = self.solution_init()
        local_obj = self.cal_objective(local_solution)
        self.best_solution = local_solution.copy()
        self.best_obj = local_obj
        neighbours = self.get_neighbours(local_solution)
        temperature = self.max_temp
        for step in trange(self.iter_num):
            # update temperature
            temperature = self.temperature_update(temperature, step)

            ni = self.choose_neighbour(neighbours)
            cur_solution = neighbours[ni]
            cur_obj = self.cal_objective(cur_solution)

            # obj: minimize the total distance 
            if cur_obj < local_obj or \
               np.random.random() < self.SA_accept(cur_obj-local_obj, temperature):
                local_solution = cur_solution
                local_obj = cur_obj
                neighbours = self.get_neighbours(local_solution)
            else:
                neighbours.pop(ni)
                if len(neighbours) == 0:
                    print('local optimal, break out, iterated {} times'.format(step))
                    break

            if local_obj < self.best_obj:
                self.best_obj = local_obj
                self.best_solution = local_solution

            self.process.append([local_obj, self.best_obj])

        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes, self.best_obj

# variable neighbourhood search algorithm
class Varialbe_Neighbourhood_Search(Heuristic):
    def __init__(self, problem, iter_num):
        """ 
        read data and preprocess 
        """
        self.read_data(problem)
        self.iter_num = iter_num

        self.choose_neighbour_strategy == "random"

        # set VNS paraments
        self.operators_list = [Reverse(), Relocate(), Exchange()]
        for k in range(3, 10):
            self.operators_list.append(Relocate(k))
            self.operators_list.append(Exchange(k))
    
    def run(self):
        best_solution = self.solution_init() # solution in form of routes
        best_obj = self.cal_objective(best_solution)
        neighbours = self.get_neighbours(best_solution, operator=self.operators_list[0])
        operator_k = 0
        for step in trange(self.iter_num):
            ni = self.choose_neighbour(neighbours)
            cur_solution = neighbours[ni]
            cur_obj = self.cal_objective(cur_solution)
            # obj: minimize the total distance 
            if cur_obj < best_obj: 
                # self.operators_list.insert(0, self.operators_list.pop(operator_k))
                operator_k = 0
                best_solution = cur_solution
                best_obj = cur_obj
                neighbours = self.get_neighbours(best_solution, operator=self.operators_list[0])
            else:
                neighbours.pop(ni)
                if len(neighbours) == 0: # when the neighbour space empty, change anothor neighbour structure(operator)
                    operator_k += 1
                    if operator_k < len(self.operators_list):
                        operator = self.operators_list[operator_k]
                        neighbours = self.get_neighbours(best_solution, operator=operator)
                    else:
                        print('local optimal, break out, iterated {} times'.format(step))
                        break

            self.process.append(best_obj)
        self.best_solution = best_solution
        self.best_obj = best_obj
        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes, self.best_obj

# variable neighbourhood search algorithm with tabu list
class VNS_tabu(Heuristic):
    def __init__(self, problem, iter_num):
        """ 
        read data and preprocess 
        """
        self.read_data(problem)
        self.iter_num = iter_num

        # set VNS paraments
        self.operators_list = [Reverse(), Relocate(), Exchange(), Relocate(5)]

        # tabu paraments
        self.tabu = []
        self.tabu_length = self.p_num

    def tabu_neighbours(self, neighbours):
        ni = 0
        while ni < len(neighbours):
            neighbour = neighbours[ni]
            if neighbour in self.tabu:
                neighbours.pop(ni)
            else:
                ni += 1
        return neighbours

    def run(self):
        local_solution = self.solution_init() # solution in form of routes
        local_obj = self.cal_objective(local_solution)
        self.best_solution = local_solution.copy()
        self.best_obj = local_obj
        neighbours = self.get_neighbours(local_solution, operator=self.operators_list[0])
        neighbours = self.tabu_neighbours(neighbours)
        operator_k = 0
        for step in trange(self.iter_num):
            ni = self.choose_neighbour(neighbours)
            cur_solution = neighbours[ni]
            cur_obj = self.cal_objective(cur_solution)
            # obj: minimize the total distance 
            if cur_obj < local_obj: 
                operator_k = 0
                local_solution = cur_solution
                local_obj = cur_obj
                neighbours = self.get_neighbours(local_solution, operator=self.operators_list[0])
                neighbours = self.tabu_neighbours(neighbours)
            else:
                throw = neighbours.pop(ni) # save throw for tabu
                if len(neighbours) == 0: # when the neighbour space empty, change anothor neighbour structure(operator)
                    operator_k += 1
                    if operator_k < len(self.operators_list):
                        operator = self.operators_list[operator_k]
                        neighbours = self.get_neighbours(local_solution, operator=operator)
                        neighbours = self.tabu_neighbours(neighbours)
                    else: # not break out, but add in tabu
                        self.tabu.append(local_solution)                        
                        if len(self.tabu) > self.tabu_length:
                            self.tabu.pop(0)
                        local_solution = throw
                        local_obj = cur_obj
                        neighbours = self.get_neighbours(local_solution, operator=self.operators_list[0])
                        neighbours = self.tabu_neighbours(neighbours)

            if local_obj < self.best_obj:
                self.best_obj = local_obj
                self.best_solution = local_solution.copy()

            self.process.append([local_obj, self.best_obj])
        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes, self.best_obj

# adaptive large neighbourhood search algorithm
class ALNS(Heuristic):
    def __init__(self, problem, iter_num):
        """ 
        read data and preprocess 
        """
        self.read_data(problem)
        self.iter_num = iter_num

        self.choose_neighbour_strategy == "first"

        # set VNS paraments
        self.operators_list = [Reverse(), Relocate(), Exchange()]
        for k in range(3, 6):
            self.operators_list.append(Relocate(k))
            self.operators_list.append(Exchange(k))
        self.operators_scores = np.ones(len(self.operators_list))
        self.operators_steps = np.ones(len(self.operators_list))
        self.adaptive_period = 10000
        self.sigma1 = 2
        self.sigma2 = 1
        self.sigma3 = 0.1
        # set paraments of SA
        self.max_temp = 1
        self.min_temp = 0
        self.a = 0.9998 
        self.a_steps = 100

    def SA_accept(self, detaC, temperature):
        return math.exp(-detaC / temperature)

    def temperature_update(self, temperature, step):
        if step % self.a_steps == 0: # update temperature by static steps
            temperature *= self.a
        temperature = max(self.min_temp, temperature)
        return temperature

    def choose_operator(self):
        weights = self.operators_scores / self.operators_steps
        prob = weights / sum(weights)
        return np.random.choice(range(len(self.operators_list)), p=prob)
    
    def get_neighbour(self, solution, operator):
        return operator.get(solution)

    def run(self):
        cur_solution = self.solution_init() # solution in form of routes
        cur_obj = self.cal_objective(cur_solution)
        self.best_solution = cur_solution
        self.best_obj = cur_obj
        temperature = self.max_temp
        for step in trange(self.iter_num):
            opt_i = self.choose_operator()
            new_solution = self.get_neighbour(cur_solution, self.operators_list[opt_i])
            new_obj = self.cal_objective(new_solution)
            # obj: minimize the total distance 
            if new_obj < self.best_obj:
                self.best_solution = new_solution
                self.best_obj = new_obj
                cur_solution = new_solution
                cur_obj = new_obj
                self.operators_scores[opt_i] += self.sigma1
                self.operators_steps[opt_i] += 1
            elif new_obj < cur_obj: 
                cur_solution = new_solution
                cur_obj = new_obj
                self.operators_scores[opt_i] += self.sigma2
                self.operators_steps[opt_i] += 1
            elif np.random.random() < self.SA_accept(new_obj-cur_obj, temperature):
                cur_solution = new_solution
                cur_obj = new_obj
                self.operators_scores[opt_i] += self.sigma3
                self.operators_steps[opt_i] += 1
            # reset operators weights
            if step % self.adaptive_period == 0: 
                self.operators_scores = np.ones(len(self.operators_list))
                self.operators_steps = np.ones(len(self.operators_list))
            # update SA temperature
            temperature = self.temperature_update(temperature, step)
            # record process obj
            self.process.append(cur_obj)
        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes, self.best_obj
    
if __name__ == "__main__":
    file_name = "Research\LinearProgramming\solomon_100\C101.txt"
    problem = read_data(file_name)
    iter_num = 1000000
    # alg = Heuristic(problem, iter_num)
    # alg = Tabu_Search(problem, iter_num)
    # alg = Simulated_Annealing(problem, iter_num)
    alg = Varialbe_Neighbourhood_Search(problem, iter_num)
    # alg = VNS_tabu(problem, iter_num)
    # alg = ALNS(problem, iter_num)
    routes, obj = alg.run()
    obj = evaluate(problem, routes)
    print('obj: {}, {} vehicles in total'.format(obj, len(routes)))
    alg.draw(routes)
    alg.show_process()