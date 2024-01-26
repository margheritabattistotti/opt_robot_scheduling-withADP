# APPROXIMATE POLICY ITERATION

import pickle as pkl
import os
import API_BVFA_simulator as sim
import API_agent as ag
import timeit
import time
from tqdm import tqdm
import random


# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

# nodes              = input_data["nodes"][:8]            # list of nodes where actions can be performed
nodes              = ['np0', 'np1', 'np2', 'nt0', 'nt1', 'nt2']  # for Large instance I
# nodes              = ['np0', 'np1', 'np2', 'nt0']       # for MINI instance
# nodes              = ['np0', 'np1', 'np2', 'nt0', 'nt1']       # for instance for Graphs
# nodes              = ['np0', 'np1', 'nt0', 'nt1']       # for instance for Graphs II
nodes_coordinates  = input_data["nodes_coordinates"]    # dictionary n:c, where n-> node, c-> (x,y) coordinates
# take [:3] for MINI instance and instance for graphs, [:2] for instance for graphs II
objects            = input_data["objects"][:3]              # list of objects that can be picked
O                  = len(objects)                       # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
# take [:1] for MINI instance
trays              = input_data["trays"]                # list of target trays where objects can be place
trays.append('tray2')                                   # for LARGE instance, need a third tray
T                  = len(trays)                         # total number of trays
trays_coordinates  = input_data["trays_coordinates"]    # dictionary t:c, where t-> tray, c-> (x,y) coordinates
trays_coordinates['tray2'] = (131.75, 148.0)            # for LARGE instance, need a third tray
nodes_connections  = input_data["nodes_connections"]    # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

Thorizon = 360  # (s) horizon of the optimization
# throwing_nodes = nodes[5:8]  # for LARGE instance
# throwing_nodes = nodes[5:7]  # for TEST instance
# throwing_nodes = [nodes[-1]]  # for MINI instance
# throwing_nodes = nodes[3:5]  # for instance for Graph
# throwing_nodes = nodes[2:4]  # for instance for Graph II
throwing_nodes = nodes[3:6]  # for instance for LARGE I

rewards = {'move': 0, 'pick': 10, 'throw': 12}


# ############################################# MISSION DEFINITION ############################################# #

# LARGE instance I (previously Large2)
mission = {"tray0": {"objectA": 3, "objectB": 2, "objectC": 2},
           "tray1": {"objectA": 0, "objectB": 2, "objectC": 4},
           "tray2": {"objectA": 1, "objectB": 4, "objectC": 0}}

# # LARGE instance II
# mission = {"tray0": {"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 1, "objectE": 2},
#            "tray1": {"objectA": 0, "objectB": 0, "objectC": 3, "objectD": 2, "objectE": 0},
#            "tray2": {"objectA": 2, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 1}}

# # TEST instance
# mission = {"tray0": {"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 0},
#            "tray1": {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 3, "objectE": 0}}

# # MINI instance
# mission = {"tray0":  {"objectA": 3, "objectB": 2, "objectC": 2}}

# # Instance for Graphs
# mission = {"tray0": {"objectA": 3, "objectB": 1, "objectC": 2},
#            "tray1": {"objectA": 2, "objectB": 2, "objectC": 1}}

# # Instance for Graphs II
# mission = {"tray0": {"objectA": 4, "objectB": 4},
#            "tray1": {"objectA": 4, "objectB": 4}}


# ################################################# MAIN EXECTUTION ################################################# #

start = timeit.default_timer()
time.sleep(1)

initial_state = [0] * (O * (T + 1) + 2)
initial_state[1] = 'np0'

discount_factor = 0.99
parameters = [0] * (3 + len(initial_state) - 1 + len(nodes) + len(throwing_nodes) + len(objects))
# That is 3 "known terms" depending on discretized time + state entries (except the position that is codified by)
# + one hot encoding for nodes (len(nodes) parameters) + parameters for phi_4 and phi_5 respectively

# # Load of parameters
# parameters = pkl.load(open(dir_path + "//API_paramsM2.pkl", "rb"))

#                        ############ Simulator and Agent objects initialization ############                        #

SIM = sim.Simulator(initial_state, nodes, nodes_coordinates, throwing_nodes, objects, objects_pick_nodes, trays,
                    trays_coordinates, nodes_connections, mission, Thorizon)

greedy_epsilon = 0.9
AG = ag.Approximator(initial_state, nodes, nodes_coordinates, throwing_nodes, objects, objects_pick_nodes, trays,
                     trays_coordinates, nodes_connections, mission, Thorizon, greedy_epsilon,
                     discount_factor, parameters)

#                       ######################## Approximation phase ########################                        #

for iters in tqdm(range(2000)):
    for m in range(10):
        states = [SIM.initial_state]
        actions = []
        # Forward simulations
        while not SIM.terminal:
            _, best_action_type, best_action, _, p_d_state = AG.get_action()
            actions.append(best_action_type)
            if best_action_type is None:
                actions.remove(actions[-1])
                break
            SIM.post_decision_state = p_d_state.copy()
            _, _, _, _, _, next_state, _, _ = SIM.simulation(best_action_type, best_action)
            AG.current_state = next_state.copy()
            states.append(next_state)

        # Double/Backward pass
        v = [SIM.terminal_value()] * (len(actions) + 1)
        for t in range(len(actions)-1, -1, -1):
            v[t] = rewards[actions[t]] * ((Thorizon * 2 - states[t][0]) / Thorizon) + discount_factor * v[t+1]
            AG.model.model_update(states[t], v[t])  # Updating of model parameters through Recursive Least Squares

        # Reset initial state for the next simulation
        SIM.reset()
        AG.reset()

    AG.model.old_thetas = AG.model.new_thetas.copy()  # Save new model parameters
    AG.epsilon *= 0.995

    SIM.initial_state[1] = random.choice(nodes)  # For robustness: set a random initial position for the initial state
    AG.initial_state = SIM.initial_state.copy()
    SIM.reset()
    AG.reset()


end = timeit.default_timer()
print(f"Time taken is {end - start}s")


# # Save model parameters
# params = AG.model.new_thetas
# with open('API_paramsM2.pkl', 'wb') as fp:
#     pkl.dump(params, fp)
# print(params)


#                       ######################## Final forward pass ########################                        #

objective_value = [0] * 30
for k in range(30):

    AG.epsilon = 0  # No more exploration, just exploitation while finally making decisions
    SIM.initial_state[1] = 'np0'
    AG.initial_state[1] = 'np0'
    SIM.reset()
    AG.reset()

    state_sequence = [initial_state]
    action_sequence = []
    while not SIM.terminal:
        _, best_action_type, best_action, _, p_d_state = AG.get_action()
        if best_action_type is None:
            break
        action_string = best_action_type + ' ' + str(best_action)
        SIM.post_decision_state = p_d_state.copy()
        _, _, _, _, failure, next_state, r, _ = SIM.simulation(best_action_type, best_action)
        if failure:
            action_string = action_string + ' ' + '(FAILURE)'
        action_sequence.append(action_string)
        state_sequence.append(next_state)
        AG.current_state = next_state.copy()
    print(action_sequence)
    print(state_sequence)
    objective_value[k] += SIM.terminal_value()

# print(objective_value)

# end = timeit.default_timer()
# print(f"Time taken is {end - start}s")

mean_obj_value = sum(objective_value)/len(objective_value)

print('Mean terminal state value: ' + str(mean_obj_value))

