# read data from dateset

import numpy as np

class problem():
    vehicle_num = 0
    vehicle_capacity = 0
    customers = np.zeros((2,2))

def read_data(file_name):
    """
    read VRPTW data from dataset
    input: file_name
    output: problem obj (including (int)vehicle_number, (int)vehicle_capacity, (numpy-array[25, 6])customers)
            ps:customers include x, y, demand, ready_time, due_time, service_time
    """
    prob = problem()

    with open(file_name) as file_object:
        lines = file_object.readlines()
    
    # load vehicle setting
    vehicle = list(map(int, lines[4].split()))
    prob.vehicle_num, prob.vehicle_capacity = vehicle

    # load customers setting
    customers = []
    for line in lines[9:]:
        cust = list(map(int, line.split()))
        customers.append(cust[1:])
    customers = np.array(customers, dtype=list) 
    prob.customers = customers

    return prob
        

if __name__ == "__main__":
    file_name = "dataset/C101.txt"
    prob = read_data(file_name)
