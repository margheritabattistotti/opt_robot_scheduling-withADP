from numpy import sort

# import A_VFA_agent as ag
import API_basis_functions_approximator as bf
import API_agent as ag
import torch
import pickle as pkl
import os
import random
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

# %%
dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

# %% extract input data
# nodes              = input_data["nodes"][:7]            # list of nodes where actions can be performed
# nodes              = ['np0', 'np1', 'np2', 'nt0']       # for MINI instance
# nodes              = ['np0', 'np1', 'np2', 'nt0', 'nt1']       # for instance for Graphs I
nodes              = ['np0', 'np1', 'nt0', 'nt1']       # for instance for Graphs II
nodes_coordinates  = input_data["nodes_coordinates"]    # dictionary n:c, where n-> node, c-> (x,y) coordinates
# take [:2] for instance for graph II, [:3] for I
objects            = input_data["objects"][:2]              # list of objects that can be picked
O                  = len(objects)                       # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
# take [:1] for MINI instance
trays              = input_data["trays"]                # list of target trays where objects can be place
T                  = len(trays)                         # total number of trays
trays_coordinates  = input_data["trays_coordinates"]    # dictionary t:c, where t-> tray, c-> (x,y) coordinates
nodes_connections  = input_data["nodes_connections"]    # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0

# %% optimization parameters & functions
maxPortableObjs = 4
Thorizon = 300  # (s) horizon of the optimization
# throwing_nodes = nodes[5:7]
# throwing_nodes = [nodes[-1]]  # for MINI instance
# throwing_nodes = nodes[3:5]  # for instance for Graph
throwing_nodes = nodes[2:4]  # for instance for Graph II

# # TEST instance
# mission = {"tray0": {"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 0},
#            "tray1": {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 3, "objectE": 0}}

# # MINI instance
# mission = {"tray0":  {"objectA": 3, "objectB": 2, "objectC": 2}}

# # Instance for Graphs
# mission = {"tray0": {"objectA": 3, "objectB": 1, "objectC": 2},
#            "tray1": {"objectA": 2, "objectB": 2, "objectC": 1}}

# Instance for Graphs II
mission = {"tray0": {"objectA": 4, "objectB": 4},
           "tray1": {"objectA": 4, "objectB": 4}}

# ############################################## EXACT VALUES LOADING ############################################## #

exactV = pkl.load(open(dir_path + "//valueF_data_mission2_TF3.pkl", "rb"))

# ########################################### APPROXIMATE VALUES LOADING ########################################### #

initial_state = [0] * (O * (T + 1) + 2)
initial_state[1] = 'np0'
greedy_epsilon = 0.5
discount_factor = 0.9

# AG = ag.Approximator(initial_state, nodes, nodes_coordinates, throwing_nodes, objects, objects_pick_nodes, trays,
#                      trays_coordinates, nodes_connections, mission, Thorizon, greedy_epsilon, discount_factor)
# AG.model.load_state_dict(torch.load('VFA_model_mission2'))

# parameters = pkl.load(open(dir_path + "//API_paramsTEST.pkl", "rb"))
parameters = pkl.load(open(dir_path + "//API_paramsM2.pkl", "rb"))
AG = ag.Approximator(initial_state, nodes, nodes_coordinates, throwing_nodes, objects, objects_pick_nodes, trays,
                     trays_coordinates, nodes_connections, mission, Thorizon, greedy_epsilon,
                     discount_factor, parameters)


# ######################################### RANDOM OVERALL COMPARIOSN PLOT ######################################### #

eV = []
aV = []

x = range(100)
states = []
for i in x:
    state = random.choice(list(exactV.keys()))
    states.append(state)
    eV.append(exactV[state])
    # aV.append(np.float64(AG.get_value(list(state))))
    aV.append(np.float64(AG.model.get_value(list(state))))

# M1
# opt_path = [[0, 'np0', 0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 'np0', 1, 0, 0, 0, 0, 0, 0, 0, 0], [14, 'np0', 2, 0, 0, 0, 0, 0, 0, 0, 0], [21, 'np0', 3, 0, 0, 0, 0, 0, 0, 0, 0], [22, 'np1', 3, 0, 0, 0, 0, 0, 0, 0, 0], [29, 'np1', 3, 0, 0, 1, 0, 0, 0, 0, 0], [39, 'nt0', 3, 0, 0, 1, 0, 0, 0, 0, 0], [44, 'nt0', 3, 0, 0, 1, 0, 1, 0, 0, 0], [49, 'nt0', 3, 1, 0, 1, 0, 1, 0, 0, 0], [54, 'nt0', 3, 2, 0, 1, 0, 1, 0, 0, 0], [59, 'nt0', 3, 3, 0, 1, 0, 1, 0, 0, 0], [64, 'np2', 3, 3, 0, 1, 0, 1, 0, 0, 0], [71, 'np2', 3, 3, 0, 1, 0, 1, 1, 0, 0], [78, 'np2', 3, 3, 0, 1, 0, 1, 2, 0, 0], [85, 'np2', 3, 3, 0, 1, 0, 1, 3, 0, 0], [86, 'np1', 3, 3, 0, 1, 0, 1, 3, 0, 0], [93, 'np1', 3, 3, 0, 2, 0, 1, 3, 0, 0], [98, 'nt0', 3, 3, 0, 2, 0, 1, 3, 0, 0], [103, 'nt0', 3, 3, 0, 2, 1, 1, 3, 0, 0], [108, 'nt0', 3, 3, 0, 2, 1, 1, 3, 1, 0], [113, 'nt0', 3, 3, 0, 2, 1, 1, 3, 2, 0], [123, 'np1', 3, 3, 0, 2, 1, 1, 3, 2, 0], [130, 'np1', 3, 3, 0, 3, 1, 1, 3, 2, 0], [131, 'np0', 3, 3, 0, 3, 1, 1, 3, 2, 0], [138, 'np0', 4, 3, 0, 3, 1, 1, 3, 2, 0], [145, 'np0', 5, 3, 0, 3, 1, 1, 3, 2, 0], [153, 'nt1', 5, 3, 0, 3, 1, 1, 3, 2, 0], [158, 'nt1', 5, 3, 1, 3, 1, 1, 3, 2, 0], [163, 'nt1', 5, 3, 2, 3, 1, 1, 3, 2, 0], [168, 'nt1', 5, 3, 2, 3, 1, 2, 3, 2, 0], [173, 'nt1', 5, 3, 2, 3, 1, 2, 3, 2, 1]]
# adp_path = [[0, 'np0', 0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 'np0', 1, 0, 0, 0, 0, 0, 0, 0, 0], [14, 'np0', 2, 0, 0, 0, 0, 0, 0, 0, 0], [21, 'np0', 3, 0, 0, 0, 0, 0, 0, 0, 0], [28, 'np0', 4, 0, 0, 0, 0, 0, 0, 0, 0], [39, 'nt0', 4, 0, 0, 0, 0, 0, 0, 0, 0], [44, 'nt0', 4, 1, 0, 0, 0, 0, 0, 0, 0], [49, 'nt0', 4, 2, 0, 0, 0, 0, 0, 0, 0], [54, 'nt0', 4, 3, 0, 0, 0, 0, 0, 0, 0], [59, 'nt0', 4, 3, 1, 0, 0, 0, 0, 0, 0], [65, 'np0', 4, 3, 1, 0, 0, 0, 0, 0, 0], [72, 'np0', 5, 3, 1, 0, 0, 0, 0, 0, 0], [73, 'np1', 5, 3, 1, 0, 0, 0, 0, 0, 0], [80, 'np1', 5, 3, 1, 1, 0, 0, 0, 0, 0], [87, 'np1', 5, 3, 1, 2, 0, 0, 0, 0, 0], [94, 'np1', 5, 3, 1, 3, 0, 0, 0, 0, 0], [99, 'nt0', 5, 3, 1, 3, 0, 0, 0, 0, 0], [104, 'nt0', 5, 3, 1, 3, 0, 1, 0, 0, 0], [109, 'nt0', 5, 3, 1, 3, 0, 2, 0, 0, 0], [114, 'nt0', 5, 3, 1, 3, 1, 2, 0, 0, 0], [119, 'nt0', 5, 3, 2, 3, 1, 2, 0, 0, 0], [124, 'np2', 5, 3, 2, 3, 1, 2, 0, 0, 0], [131, 'np2', 5, 3, 2, 3, 1, 2, 1, 0, 0], [138, 'np2', 5, 3, 2, 3, 1, 2, 2, 0, 0], [145, 'np2', 5, 3, 2, 3, 1, 2, 3, 0, 0], [150, 'nt0', 5, 3, 2, 3, 1, 2, 3, 0, 0], [155, 'nt0', 5, 3, 2, 3, 1, 2, 3, 1, 0], [160, 'nt0', 5, 3, 2, 3, 1, 2, 3, 2, 0], [165, 'nt0', 5, 3, 2, 3, 1, 2, 3, 2, 1]]
# M2
# opt_path = [[0, 'np0', 0, 0, 0, 0, 0, 0], [7, 'np0', 1, 0, 0, 0, 0, 0], [14, 'np0', 2, 0, 0, 0, 0, 0], [21, 'np0', 3, 0, 0, 0, 0, 0], [28, 'np0', 4, 0, 0, 0, 0, 0], [34, 'nt0', 4, 0, 0, 0, 0, 0], [39, 'nt0', 4, 1, 0, 0, 0, 0], [44, 'nt0', 4, 2, 0, 0, 0, 0], [49, 'nt0', 4, 3, 0, 0, 0, 0], [54, 'nt0', 4, 4, 0, 0, 0, 0], [59, 'np1', 4, 4, 0, 0, 0, 0], [66, 'np1', 4, 4, 0, 1, 0, 0], [73, 'np1', 4, 4, 0, 2, 0, 0], [80, 'np1', 4, 4, 0, 3, 0, 0], [87, 'np1', 4, 4, 0, 4, 0, 0], [92, 'nt0', 4, 4, 0, 4, 0, 0], [97, 'nt0', 4, 4, 0, 4, 1, 0], [102, 'nt0', 4, 4, 0, 4, 2, 0], [107, 'nt0', 4, 4, 0, 4, 3, 0], [112, 'nt0', 4, 4, 0, 4, 4, 0], [117, 'np1', 4, 4, 0, 4, 4, 0], [124, 'np1', 4, 4, 0, 5, 4, 0], [131, 'np1', 4, 4, 0, 6, 4, 0], [138, 'np1', 4, 4, 0, 7, 4, 0], [145, 'np1', 4, 4, 0, 8, 4, 0], [153, 'nt1', 4, 4, 0, 8, 4, 0], [158, 'nt1', 4, 4, 0, 8, 4, 1], [163, 'nt1', 4, 4, 0, 8, 4, 2], [168, 'nt1', 4, 4, 0, 8, 4, 3], [173, 'nt1', 4, 4, 0, 8, 4, 4], [186, 'np0', 4, 4, 0, 8, 4, 4], [193, 'np0', 5, 4, 0, 8, 4, 4], [200, 'np0', 6, 4, 0, 8, 4, 4], [207, 'np0', 7, 4, 0, 8, 4, 4], [214, 'np0', 8, 4, 0, 8, 4, 4], [222, 'nt1', 8, 4, 0, 8, 4, 4], [227, 'nt1', 8, 4, 1, 8, 4, 4], [232, 'nt1', 8, 4, 2, 8, 4, 4], [237, 'nt1', 8, 4, 3, 8, 4, 4], [242, 'nt1', 8, 4, 4, 8, 4, 4]]
# adp_path = [[0, 'np0', 0, 0, 0, 0, 0, 0], [7, 'np0', 1, 0, 0, 0, 0, 0], [14, 'np0', 2, 0, 0, 0, 0, 0], [21, 'np0', 3, 0, 0, 0, 0, 0], [22, 'np1', 3, 0, 0, 0, 0, 0], [29, 'np1', 3, 0, 0, 1, 0, 0], [34, 'nt0', 3, 0, 0, 1, 0, 0], [39, 'nt0', 3, 0, 0, 1, 1, 0], [44, 'nt0', 3, 1, 0, 1, 1, 0], [49, 'nt0', 3, 2, 0, 1, 1, 0], [54, 'nt0', 3, 3, 0, 1, 1, 0], [59, 'np1', 3, 3, 0, 1, 1, 0], [66, 'np1', 3, 3, 0, 2, 1, 0], [73, 'np1', 3, 3, 0, 3, 1, 0], [80, 'np1', 3, 3, 0, 4, 1, 0], [87, 'np1', 3, 3, 0, 5, 1, 0], [92, 'nt0', 3, 3, 0, 5, 1, 0], [97, 'nt0', 3, 3, 0, 5, 2, 0], [102, 'nt0', 3, 3, 0, 5, 3, 0], [107, 'nt0', 3, 3, 0, 5, 4, 0], [112, 'nt0', 3, 3, 0, 5, 4, 1], [117, 'np1', 3, 3, 0, 5, 4, 1], [124, 'np1', 3, 3, 0, 6, 4, 1], [131, 'np1', 3, 3, 0, 7, 4, 1], [138, 'np1', 3, 3, 0, 8, 4, 1], [139, 'np0', 3, 3, 0, 8, 4, 1], [146, 'np0', 4, 3, 0, 8, 4, 1], [152, 'nt0', 4, 3, 0, 8, 4, 1], [157, 'nt0', 4, 4, 0, 8, 4, 1], [162, 'nt0', 4, 4, 0, 8, 4, 2], [167, 'nt0', 4, 4, 0, 8, 4, 3], [172, 'nt0', 4, 4, 0, 8, 4, 4], [178, 'np0', 4, 4, 0, 8, 4, 4], [185, 'np0', 5, 4, 0, 8, 4, 4], [192, 'np0', 6, 4, 0, 8, 4, 4], [199, 'np0', 7, 4, 0, 8, 4, 4], [206, 'np0', 8, 4, 0, 8, 4, 4], [212, 'nt0', 8, 4, 0, 8, 4, 4], [217, 'nt0', 8, 4, 1, 8, 4, 4], [222, 'nt0', 8, 4, 2, 8, 4, 4], [227, 'nt0', 8, 4, 3, 8, 4, 4], [232, 'nt0', 8, 4, 4, 8, 4, 4]]
# TEST
# opt_path = [[0, 'np0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 'np1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [8, 'np1', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [9, 'np2', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [16, 'np2', 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [23, 'np2', 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [30, 'np2', 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [35, 'nt0', 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [40, 'nt0', 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [45, 'nt0', 0, 0, 0, 1, 1, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0], [50, 'nt0', 0, 0, 0, 1, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [56, 'np0', 0, 0, 0, 1, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [63, 'np0', 1, 0, 0, 1, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [70, 'np0', 2, 0, 0, 1, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [78, 'nt1', 2, 0, 0, 1, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [83, 'nt1', 2, 0, 1, 1, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [88, 'nt1', 2, 0, 2, 1, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [94, 'np3', 2, 0, 2, 1, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [101, 'np3', 2, 0, 2, 1, 1, 0, 3, 2, 0, 1, 0, 0, 0, 0, 0], [108, 'np3', 2, 0, 2, 1, 1, 0, 3, 2, 0, 2, 0, 0, 0, 0, 0], [115, 'np3', 2, 0, 2, 1, 1, 0, 3, 2, 0, 3, 0, 0, 0, 0, 0], [121, 'nt1', 2, 0, 2, 1, 1, 0, 3, 2, 0, 3, 0, 0, 0, 0, 0], [126, 'nt1', 2, 0, 2, 1, 1, 0, 3, 2, 1, 3, 0, 0, 0, 0, 0], [131, 'nt1', 2, 0, 2, 1, 1, 0, 3, 2, 1, 3, 0, 1, 0, 0, 0], [136, 'nt1', 2, 0, 2, 1, 1, 0, 3, 2, 1, 3, 0, 2, 0, 0, 0], [141, 'nt1', 2, 0, 2, 1, 1, 0, 3, 2, 1, 3, 0, 3, 0, 0, 0]]
# adp_path = [[0, 'np0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 'np0', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [9, 'np2', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [16, 'np2', 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [23, 'np2', 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [30, 'np2', 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [40, 'nt0', 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [45, 'nt0', 1, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0], [50, 'nt0', 1, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0], [55, 'nt0', 1, 0, 0, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0], [60, 'nt0', 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0], [66, 'np3', 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0], [73, 'np3', 1, 0, 1, 0, 0, 0, 3, 2, 1, 1, 0, 0, 0, 0, 0], [80, 'np3', 1, 0, 1, 0, 0, 0, 3, 2, 1, 2, 0, 0, 0, 0, 0], [87, 'np3', 1, 0, 1, 0, 0, 0, 3, 2, 1, 3, 0, 0, 0, 0, 0], [90, 'np0', 1, 0, 1, 0, 0, 0, 3, 2, 1, 3, 0, 0, 0, 0, 0], [97, 'np0', 2, 0, 1, 0, 0, 0, 3, 2, 1, 3, 0, 0, 0, 0, 0], [103, 'nt0', 2, 0, 1, 0, 0, 0, 3, 2, 1, 3, 0, 0, 0, 0, 0], [108, 'nt0', 2, 0, 1, 0, 0, 0, 3, 2, 1, 2, 0, 0, 0, 0, 0], [113, 'nt0', 2, 0, 1, 0, 0, 0, 3, 2, 1, 2, 0, 1, 0, 0, 0], [118, 'nt0', 2, 0, 1, 0, 0, 0, 3, 2, 1, 2, 0, 2, 0, 0, 0], [123, 'nt0', 2, 0, 2, 0, 0, 0, 3, 2, 1, 2, 0, 2, 0, 0, 0], [129, 'np3', 2, 0, 2, 0, 0, 0, 3, 2, 1, 2, 0, 2, 0, 0, 0], [136, 'np3', 2, 0, 2, 0, 0, 0, 3, 2, 1, 3, 0, 2, 0, 0, 0], [138, 'np1', 2, 0, 2, 0, 0, 0, 3, 2, 1, 3, 0, 2, 0, 0, 0], [145, 'np1', 2, 0, 2, 1, 0, 0, 3, 2, 1, 3, 0, 2, 0, 0, 0], [150, 'nt0', 2, 0, 2, 1, 0, 0, 3, 2, 1, 3, 0, 2, 0, 0, 0], [155, 'nt0', 2, 0, 2, 1, 1, 0, 3, 2, 1, 3, 0, 2, 0, 0, 0], [160, 'nt0', 2, 0, 2, 1, 1, 0, 3, 2, 1, 3, 0, 3, 0, 0, 0]]
# for state in opt_path:
#     eV.append(exactV[tuple(state)])
#     # aV.append(np.float64(AG.get_value(list(state))))
#     aV.append(np.float64(AG.model.get_value(state)))
#
# x = range(len(opt_path))

# Create dataset
data = {
   'States': x,
   'Values': eV,
   'approxV': aV,
}

# Create dataframe
df = pd.DataFrame(data)

# Create Line plot
fig = px.line(df, x=df['States'], y=df['Values'])

# Add Scatter plot
fig.add_scatter(x=df['States'], y=df['approxV'])

# Display the plot
fig.show()

# ###################################### RANDOM OVERALL COMPARIOSN OF VALUES ###################################### #

# for i in x:
#     print(states[i], end=' ')
#     print(eV[i], end=' ')
#     print(aV[i])


# ##################################### FUNCTIONS NEEDED FOR STATES DEFINITION ##################################### #

def objects4mission_dict(mission_dict):
    # INPUT
    # mission_dict: mission nested dictionary as defined in main
    # OUTPUT
    # tot_object: dictionary {objectType: total amount to be placed during mission}
    obj_keys_list = list(list(mission_dict.values())[0].keys())
    tot_object = {}
    for key in obj_keys_list:
        tot_object[key] = 0
    for l in range(len(mission_dict)):
        for key in obj_keys_list:
            tot_object[key] += list(mission_dict.values())[l][key]
    return tot_object


obj4mission_dict = objects4mission_dict(mission)


def objects_placed(st):
    p = [0] * O
    for j in range(len(p)):
        for t in range(T):
            p[j] += st[3 + (T+1) * j + t]
    return p


def tot_objects_placed(st):
    return sum(objects_placed(st))


def tot_objects_picked(st):
    p = [0] * O
    for j in range(len(p)):
        for t in range(T):
            p[j] += st[2 + (T + 1) * j]
    return sum(p)


def objects_collected(st):
    # Yields the number of objects collected but not yet placed (inferred from the state values)
    # INPUT
    # s: current state of the system
    # OUTPUT
    # obj: the number of objects collected but not yet placed in the given state
    objs = 0
    p = objects_placed(st)
    for j in range(O):
        objs += st[2 + (T+1) * j] - p[j]  # picked - placed
    return objs


def state_space_brute_force(tot_obj_dict, maxObj):
    # Generates all possible final states in terms of picked and placed object types, without accounting for
    # final time and node position. Tailored for 5 object types and 2 trays.
    # INPUT
    # tot_obj_dict: {object: total amount to be placed during mission}, output of objects4mission_dict(mission)
    # maxObj: maximum number of objects to be carried at a time by the robot
    # OUTPUT
    # all_final_state: list of all possible final state configurations
    # Definition of all possible states such that picked[i]>=placed[i] for all objects i
    # # MISSION I
    # all_states = [[a1, a2, a3, b1, b2, b3, c1, c2, c3]
    #               for a1 in range(list(tot_obj_dict.values())[0] + 1)
    #               for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
    #               for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
    #               for b1 in range(list(tot_obj_dict.values())[1] + 1)
    #               for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
    #               for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)
    #               for c1 in range(list(tot_obj_dict.values())[2] + 1)
    #               for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)
    #               for c3 in range(0, list(list(mission.values())[1].values())[2] + 1)]

    # MISSION II
    all_states = [[a1, a2, a3, b1, b2, b3]
                  for a1 in range(list(tot_obj_dict.values())[0] + 1)
                  for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
                  for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
                  for b1 in range(list(tot_obj_dict.values())[1] + 1)
                  for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
                  for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)]

    # Filtering the states such that the robot does not carry more than maxObj objects
    all_states_b = []
    for st in all_states:
        carriedObj = objects_collected([0, 0] + st)
        if 0 <= carriedObj <= maxObj:
            all_states_b.append(st)

    # Filtering the states such that picked >= tot_placed
    all_states_c = []
    for st in all_states_b:
        p = objects_placed([0, 0] + st)
        check = 0
        for j in range(O):
            if st[(T + 1) * j] >= p[j]:
                check += 1
        if check == O:
            all_states_c.append(st)

    return all_states_c


states_list = state_space_brute_force(obj4mission_dict, maxPortableObjs)

# ############################################# TIME DEPENDENT GRAPHS ############################################ #
# x = range(Thorizon + 1)
# E_Values = []  # Exact Values
# A_Values = []  # Approximate Values
# fixed_states = []
# for g in range(10):
#     fixed_state = random.choice(states_list)
#     fixed_node = random.choice(nodes)
#     fixed_states.append([fixed_node] + fixed_state)
#
#     E_Values.append([])
#     A_Values.append([])
#
#     for t in x:
#         s = [t] + fixed_states[g]
#         E_Values[g].append(exactV[tuple(s)])  # / 10 for resize
#         A_Values[g].append(np.float64(AG.get_value(list(s))))
#
#
# fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
# fig.suptitle('Time-dependence of Value Functions', x=0.51)
# fig.set_figwidth(10)
#
# for row in range(2):
#     for col in range(5):
#         axs[row, col].plot(x, E_Values[col + 5 * row])
#         # axs[row, col].plot(x, A_Values[col + 5 * row], color='red')
#         axs[row, col].grid()
#
# axs[0, 0].set(ylabel='Values')
# axs[1, 0].set(ylabel='Values')
# axs[1, 2].set(xlabel='Time (s)')
# plt.subplots_adjust(hspace=0.15)
# plt.subplots_adjust(wspace=0.15)
# plt.show()


# ############################################# NODE DEPENDENT GRAPHS ############################################ #
# x = nodes.copy()
# E_Values = []
# A_Values = []
# fixed_states = []
# for g in range(10):
#     fixed_state = random.choice(states_list)
#     fixed_time = random.randint(0, Thorizon)
#     fixed_states.append([fixed_time] + [0] + fixed_state)
#
#     E_Values.append([])
#     A_Values.append([])
#
#     for n in x:
#         fixed_states[g][1] = n
#         s = fixed_states[g]
#         E_Values[g].append(exactV[tuple(s)])  # /10 for resize
#         A_Values[g].append(np.float64(AG.get_value(list(s))))
#
# fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
# fig.suptitle('Nodes-dependence of Value Functions', x=0.51)
# fig.set_figwidth(10)
#
# for row in range(2):
#     for col in range(5):
#         axs[row, col].plot(x, E_Values[col + 5 * row])
#         # axs[row, col].plot(x, A_Values[col + 5 * row], color='red')
#         axs[row, col].grid()
#
# axs[0, 0].set(ylabel='Values')
# axs[1, 0].set(ylabel='Values')
# axs[1, 2].set(xlabel='Nodes')
# plt.subplots_adjust(hspace=0.15)
# plt.subplots_adjust(wspace=0.15)
# plt.show()

# ########################################### PICKED OBJ DEPENDENT GRAPHS ########################################## #
# obj_types = ['A', 'B', 'C']
# # obj = 0  # Objects A
# obj = 1  # Objects B
# # obj = 2  # Object C
#
# required = obj4mission_dict[objects[obj]]
# x = []
# E_Values = []
# A_Values = []
# fixed_states = []
# for g in range(10):
#     x.append([])
#     while len(x[g]) == 0:
#         x[g] = list(range(required + 1))
#         fixed_state = random.choice(states_list)
#         fixed_time = random.randint(0, Thorizon)
#         fixed_node = random.choice(nodes)
#         fixed_states.append([fixed_time] + [fixed_node] + fixed_state)
#
#         E_Values.append([])
#         A_Values.append([])
#
#         for q in range(required+1):
#             fixed_states[g][2 + (T+1) * obj] = q
#             placed = objects_placed(fixed_states[g])[obj]
#             collected = objects_collected(fixed_states[g])
#             if q >= placed and collected <= maxPortableObjs:
#                 s = fixed_states[g]
#                 E_Values[g].append(exactV[tuple(s)])  # / 10 for resize
#                 A_Values[g].append(np.float64(AG.get_value(list(s))))
#             else:
#                 x[g].remove(q)
#
# fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
# fig.suptitle('Value Functions dependence on ' + obj_types[obj] + '-type objects picked', x=0.51)
# fig.set_figwidth(10)
#
# for row in range(2):
#     for col in range(5):
#         axs[row, col].plot(x[col + 5 * row], E_Values[col + 5 * row])
#         # axs[row, col].plot(x[col + 5 * row], A_Values[col + 5 * row], color='red')
#         axs[row, col].grid()
#
# axs[0, 0].set(ylabel='Values')
# axs[1, 0].set(ylabel='Values')
# axs[1, 2].set(xlabel='# ' + obj_types[obj] + ' objects picked')
# plt.subplots_adjust(hspace=0.15)
# plt.subplots_adjust(wspace=0.15)
# plt.show()

# ########################################### PLACED OBJ DEPENDENT GRAPHS ########################################## #
# obj_types = ['A', 'B', 'C']
# # obj = 0  # Objects A
# obj = 1  # Objects B
# # obj = 2  # Object C
#
# tr = 0
# # tr = 1
#
# required = mission[trays[tr]][objects[obj]]
# x = []
# E_Values = []
# A_Values = []
# fixed_states = []
# for g in range(10):
#     x.append([])
#     while len(x[g]) <= 0:
#         x[g] = list(range(required + 1))
#         fixed_state = random.choice(states_list)
#         fixed_time = random.randint(0, Thorizon)
#         fixed_node = random.choice(nodes)
#         fixed_states.append([fixed_time] + [fixed_node] + fixed_state)
#
#         E_Values.append([])
#         A_Values.append([])
#
#         for q in range(required+1):
#             fixed_states[g][3 + (T+1) * obj + tr] = q
#             picked_4_tr = fixed_states[g][2 + (T+1) * obj] - fixed_states[g][3 + (T+1) * obj + tr - 1 * tr + (1-tr)]
#             collected = objects_collected(fixed_states[g])
#             if q <= picked_4_tr and collected <= maxPortableObjs:
#                 s = fixed_states[g]
#                 E_Values[g].append(exactV[tuple(s)])  # /10 for resize
#                 A_Values[g].append(np.float64(AG.get_value(list(s))))
#             else:
#                 x[g].remove(q)
#
# fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
# fig.suptitle('Value Functions dependence on ' + obj_types[obj] + '-type objects placed in tray ' + str(tr), x=0.51)
# fig.set_figwidth(10)
#
# for row in range(2):
#     for col in range(5):
#         axs[row, col].plot(x[col + 5 * row], E_Values[col + 5 * row])
#         # axs[row, col].plot(x[col + 5 * row], A_Values[col + 5 * row], color='red')
#         axs[row, col].grid()
#
# axs[0, 0].set(ylabel='Values')
# axs[1, 0].set(ylabel='Values')
# axs[1, 2].set(xlabel='# ' + obj_types[obj] + ' objects placed in tray ' + str(tr))
# plt.subplots_adjust(hspace=0.15)
# plt.subplots_adjust(wspace=0.15)
# plt.show()

# ######################################## SEEK FOR MORE INFORMATIVE GRAPHS ######################################## #

# ############################## NODES-Dependence when Collected = MaxPortableObjects ############################# #
# x = nodes.copy()
# E_Values = []
# A_Values = []
# fixed_states = []
# for g in range(10):
#     collected = 0
#     while collected != maxPortableObjs:
#         fixed_state = random.choice(states_list)
#         fixed_time = random.randint(0, Thorizon)
#         fixed_state = [fixed_time] + [0] + fixed_state
#         collected = objects_collected(fixed_state)
#
#     fixed_states.append(fixed_state)
#
#     E_Values.append([])
#     # A_Values.append([])
#
#     for n in x:
#         fixed_states[g][1] = n
#         s = fixed_states[g]
#         E_Values[g].append(exactV[tuple(s)])  # /10 for resize
#         # A_Values[g].append(np.float64(AG.get_value(list(s))))
#
# fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
# fig.suptitle('Nodes-dependence of Value Functions when Robot is carrying the maximum number of objects', x=0.51)
# fig.set_figwidth(6)
#
# for row in range(2):
#     for col in range(3):
#         axs[row, col].plot(x, E_Values[col + 3 * row])
#         # axs[row, col].plot(x, A_Values[col + 3 * row], color='red')
#         axs[row, col].grid()
#
# axs[0, 0].set(ylabel='Values')
# axs[1, 0].set(ylabel='Values')
# axs[1, 1].set(xlabel='Nodes')
# plt.subplots_adjust(hspace=0.15)
# plt.subplots_adjust(wspace=0.15)
# plt.show()


# ##################################### NODES-Dependence when Collected == 0 #################################### #
# x = nodes.copy()
# E_Values = []
# # A_Values = []
# fixed_states = []
# for g in range(10):
#     collected = 10
#     while collected != 0:
#         fixed_state = random.choice(states_list)
#         fixed_time = random.randint(0, Thorizon)
#         fixed_state = [fixed_time] + [0] + fixed_state
#         collected = objects_collected(fixed_state)
#
#     fixed_states.append(fixed_state)
#
#     E_Values.append([])
#     # A_Values.append([])
#
#     for n in x:
#         fixed_states[g][1] = n
#         s = fixed_states[g]
#         E_Values[g].append(exactV[tuple(s)])  # /10 for resize
#         # A_Values[g].append(np.float64(AG.get_value(list(s))))
#
# fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
# fig.suptitle('Nodes-dependence of Value Functions when Robot is carrying no objects', x=0.51)
# fig.set_figwidth(6)
#
# for row in range(2):
#     for col in range(3):
#         axs[row, col].plot(x, E_Values[col + 3 * row])
#         # axs[row, col].plot(x, A_Values[col + 3 * row], color='red')
#         axs[row, col].grid()
#
# axs[0, 0].set(ylabel='Values')
# axs[1, 0].set(ylabel='Values')
# axs[1, 1].set(xlabel='Nodes')
# plt.subplots_adjust(hspace=0.15)
# plt.subplots_adjust(wspace=0.15)
# plt.show()

# ################################ NODES-Dependence when picked < Objects4Mission ############################### #
# ################################################# NOT USEFUL ################################################# #


# ############################ Fixed-time, fixed-node: total objects collected dependence ########################### #
# x = range(50)
# E_Values = []  # Exact Values
# A_Values = []  # Approximate Values
# fixed_states = []
# for g in range(10):
#     time = random.randint(0, Thorizon)
#     node = random.choice(nodes)
#     fixed_states.append([])
#     for s in range(50):
#         fixed_states[g].append([time] + [node] + random.choice(states_list))
#     fixed_states[g].sort(key=objects_collected)
#
#     E_Values.append([])
#     A_Values.append([])
#
#     for s in fixed_states[g]:
#         E_Values[g].append(exactV[tuple(s)])  # / 10 for resize
#         A_Values[g].append(np.float64(AG.get_value(list(s))))
#
#
# fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
# fig.suptitle('Fixed time, fixed node: total objects collected dependence', x=0.51)
# fig.set_figwidth(10)
#
# for row in range(2):
#     for col in range(5):
#         axs[row, col].scatter(x, E_Values[col + 5 * row])
#         # axs[row, col].plot(x, A_Values[col + 5 * row], color='red')
#         axs[row, col].grid()
#         axs[row, col].set_title('T= '+ str(fixed_states[col + 5 * row][0][0]) +
#                                 ', N= ' + fixed_states[col + 5 * row][0][1])
#
# axs[0, 0].set(ylabel='Values')
# axs[1, 0].set(ylabel='Values')
# axs[1, 2].set(xlabel='Total objects collected')
# plt.subplots_adjust(hspace=0.15)
# plt.subplots_adjust(wspace=0.15)
# plt.show()


# ############################ Fixed-time, fixed-node: total objects placed dependence ########################### #
# x = range(50)
# E_Values = []  # Exact Values
# A_Values = []  # Approximate Values
# fixed_states = []
# for g in range(10):
#     time = random.randint(0, Thorizon)
#     node = random.choice(nodes)
#     fixed_states.append([])
#     for s in range(50):
#         fixed_states[g].append([time] + [node] + random.choice(states_list))
#     fixed_states[g].sort(key=tot_objects_placed)
#
#     E_Values.append([])
#     A_Values.append([])
#
#     for s in fixed_states[g]:
#         E_Values[g].append(exactV[tuple(s)])  # / 10 for resize
#         A_Values[g].append(np.float64(AG.get_value(list(s))))
#
#
# fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
# fig.suptitle('Fixed time, fixed node: total objects placed dependence', x=0.51)
# fig.set_figwidth(10)
#
# for row in range(2):
#     for col in range(5):
#         axs[row, col].scatter(x, E_Values[col + 5 * row])
#         # axs[row, col].plot(x, A_Values[col + 5 * row], color='red')
#         axs[row, col].grid()
#         axs[row, col].set_title('T= '+ str(fixed_states[col + 5 * row][0][0]) +
#                                 ', N= ' + fixed_states[col + 5 * row][0][1])
#
# axs[0, 0].set(ylabel='Values')
# axs[1, 0].set(ylabel='Values')
# axs[1, 2].set(xlabel='Total objects placed')
# plt.subplots_adjust(hspace=0.15)
# plt.subplots_adjust(wspace=0.15)
# plt.show()


# ############################ Fixed-time, fixed-node: total objects picked dependence ########################### #
x = range(50)
E_Values = []  # Exact Values
# A_Values = []  # Approximate Values
fixed_states = []
for g in range(10):
    time = random.randint(0, Thorizon)
    node = random.choice(nodes)
    fixed_states.append([])
    for s in range(50):
        fixed_states[g].append([time] + [node] + random.choice(states_list))
    fixed_states[g].sort(key=tot_objects_picked)

    E_Values.append([])
    # A_Values.append([])

    for s in fixed_states[g]:
        E_Values[g].append(exactV[tuple(s)])  # / 10 for resize
        # A_Values[g].append(np.float64(AG.get_value(list(s))))


fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
fig.suptitle('Fixed time, fixed node: total objects picked dependence', x=0.51)
fig.set_figwidth(10)

for row in range(2):
    for col in range(5):
        axs[row, col].scatter(x, E_Values[col + 5 * row])
        # axs[row, col].plot(x, A_Values[col + 5 * row], color='red')
        axs[row, col].grid()
        axs[row, col].set_title('T= '+ str(fixed_states[col + 5 * row][0][0]) +
                                ', N= ' + fixed_states[col + 5 * row][0][1])

axs[0, 0].set(ylabel='Values')
axs[1, 0].set(ylabel='Values')
axs[1, 2].set(xlabel='Ranking of sorted states')
plt.subplots_adjust(hspace=0.15)
plt.subplots_adjust(wspace=0.15)
plt.show()
