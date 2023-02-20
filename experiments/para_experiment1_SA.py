# parament experiment for SA
# author: Charles Lee
# date: 2022.09.24

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import trange

from read_data import read_data
from VRP_heuristics import *
from simulated_annealing import Simulated_Annealing as SA_alg

para1_max_temp = [10, 25, 100, 1000]
para2_r = [0.9, 0.98, 0.997]
para3_r_steps = [1, 10, 40, 100]

file_name = 'dataset/C101.txt'
problem = read_data(file_name)
iter_num = 100000
experiment_num = 10
mean_obj = np.zeros((len(para1_max_temp), len(para2_r), len(para3_r_steps)))
var_obj = np.zeros((len(para1_max_temp), len(para2_r), len(para3_r_steps)))
for i1 in trange(len(para1_max_temp)):
    for i2 in range(len(para2_r)):
        for i3 in range(len(para3_r_steps)):
            obj_list = []
            for ei in range(experiment_num):
                alg = SA_alg(problem, iter_num)
                alg.max_temp = para1_max_temp[i1]
                alg.a = para2_r[i2]
                alg.a_steps = para3_r_steps[i3]
                routes, obj = alg.run()
                obj_list.append(obj)
            mean_obj[i1, i2, i3] = np.mean(obj_list)
            var_obj[i1, i2, i3] = np.var(obj_list)
np.save("experiments/SA_mean_obj.npy", mean_obj)
np.save("experiments/SA_var_obj.npy", var_obj)



