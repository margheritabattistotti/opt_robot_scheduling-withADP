import pickle as pkl
import os
import random
import timeit
import time


# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

nodes              = input_data["nodes"][:8]            # list of nodes where actions can be performed
# nodes              = input_data["nodes"][:7]            # list of nodes where actions can be performed
# nodes.remove('np4')                                     # for MEDIUM instance
# nodes              = ['np0', 'np1', 'np2', 'nt0']       # for MINI instance to compare with DANI
# nodes = ['np0', 'np1', 'np2', 'nt0', 'nt1']             # for instance for Graphs
nodes_coordinates  = input_data["nodes_coordinates"]    # dictionary n:c, where n-> node, c-> (x,y) coordinates
# take [:3] for MINI instance and instance for graphs, [:4] for MEDIUM, all for LARGE and TEST
objects            = input_data["objects"]              # list of objects that can be picked
O                  = len(objects)                       # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
trays              = input_data["trays"]                # list of target trays where objects can be place
trays.append('tray2')                                   # for LARGE instance, need a third tray
T                  = len(trays)                         # total number of trays
# T                  = 1                                  # for MINI instance
trays_coordinates  = input_data["trays_coordinates"]    # dictionary t:c, where t-> tray, c-> (x,y) coordinates
trays_coordinates['tray2'] = (131.75, 148.0)            # for LARGE instance, need a third tray
nodes_connections  = input_data["nodes_connections"]    # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

# Thorizon = 200  # (s) horizon of the optimization
# Thorizon = 120  # for MINI instance
Thorizon = 300  # (s) for LARGE instance
tpick = 7  # (s) time elapsed to perform picking action
tplace = 5  # (s) time elapsed to perform throwing action


# ############################################# MISSION DEFINITION ############################################# #

# # TEST instance
# mission = {"tray0": {"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 0},
#            "tray1": {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 3, "objectE": 0}}

# # TEST to compare with priorities

# mission = {"tray0": {"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 0, "objectE": 1},
#            "tray1": {"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 0, "objectE": 1}}

# # MINI instance to compare with DANI
# mission = {"tray0":  {"objectA": 3, "objectB": 2, "objectC": 2}}

# # Instance for Graphs
# mission = {"tray0": {"objectA": 3, "objectB": 1, "objectC": 2},
#            "tray1": {"objectA": 2, "objectB": 2, "objectC": 1}}

# # MEDIUM instance
# mission = {"tray0": {"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0},
#            "tray1": {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 3}}

# # LARGE instance
# mission = {"tray0": {"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 0, "objectE": 1},
#            "tray1": {"objectA": 0, "objectB": 2, "objectC": 1, "objectD": 3, "objectE": 0},
#            "tray2": {"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 0, "objectE": 1}}

# # LARGE instance I (previously Large2)
# mission = {"tray0": {"objectA": 3, "objectB": 2, "objectC": 2},
#            "tray1": {"objectA": 0, "objectB": 2, "objectC": 4},
#            "tray2": {"objectA": 1, "objectB": 4, "objectC": 0}}

# LARGE instance II
mission = {"tray0": {"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 1, "objectE": 2},
           "tray1": {"objectA": 0, "objectB": 0, "objectC": 3, "objectD": 2, "objectE": 0},
           "tray2": {"objectA": 2, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 1}}


# Defined by me
maxPortableObjs = 4
rewards = {'move': 0, 'pick': 10, 'place': 12}

# I can only place or throw from placing nodes (to any tray)
# placing_nodes = nodes[5:7]  # for TEST instance
placing_nodes = [nodes[-1]]  # for MINI instance to compare with DANI
# placing_nodes = nodes[4:6]  # for MEDIUM instance
# placing_nodes = nodes[5:8]  # for LARGE instance
# placing_nodes = nodes[3:5]  # instance for graphs


# ############################################# STATE SPACE DEFINITION ############################################# #

# Needed for state space computation
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


# def objects4mission_list():
#     # OUTPUT
#     # tot_object: dictionary {objectType: total amount to be placed during mission}
#     tot_object = [0] * O
#     for j in range(O):
#         for t in range(T):
#             tot_object[j] += list(mission.values())[t][objects[j]]
#     return tot_object

# obj4mission_list = objects4mission_list()


def generate_final_state(Thor, nodes_list, tot_obj_dict, num_obj, maxObj):
    # Generates a final state (time horizon reached) in terms of picked and placed object types
    # INPUT
    # Thor: time horizon
    # nodes_list: list of nodes' labels
    # tot_obj_dict: {object: total amount to be placed during mission}, output of objects4mission_dict(mission)
    # num_obj: total number of object types
    # maxObj: maximum number of objects to be carried at a time by the robot
    # OUTPUT
    # final_s: generated final state
    # tot_time_pick: total time spent picking to reach the generated final state
    # tot_time_place: total time spent placing to reach the generated final state
    final_s = [0] * (2 + (T+1) * num_obj)
    final_s[0] = Thor
    final_s[1] = random.choice(nodes_list)
    collected = 0  # collected but still not placed objects cannot be more than maxObj
    tot_time_pick = 0
    tot_time_place = 0
    for j in range(num_obj):
        picked = random.randint(0, list(tot_obj_dict.values())[j])
        placed = [0] * T
        for t in range(T):
            placed[t] = random.randint(0, list(list(mission.values())[t].values())[j])
            final_s[3 + (T+1) * j + t] = placed[t]
            tot_time_place += placed[t] * 5
        if picked - sum(placed) < 0:
            picked = sum(placed)
        collected += picked-sum(placed)
        if maxObj >= collected >= 0:
            final_s[2 + (T+1) * j] = picked
            tot_time_pick += picked * 7
        elif maxObj <= collected:
            final_s[2 + (T+1) * j] = sum(placed)
            tot_time_pick += sum(placed) * 7

    return final_s, tot_time_pick, tot_time_place


# NOT WORKING - I would like to define a state space that is independent of knowing a priori the number of object types
# def state_space(Thor, nodes_list, tot_obj_dict, num_obj):
#     # Thor: time horizon
#     # nodes_list: list of nodes' labels
#     # tot_obj_dict: {object: total amount to be placed during mission}, output of objects_mission_dict(mission)
#     # num_obj: total number of object types
#     # OUTPUT
#     # List of all possible states in the state space
#     time_node_state = [[t, n] for t in range(Thor + 1) for n in nodes_list]
#
#     pick_place_combo = {}
#     for j in range(num_obj):
#         pick_place_couples = []
#         for k in range(list(tot_obj_dict.values())[j] + 1):
#             placed = k
#             for p in range(placed + 1):
#                 picked = p
#                 pick_place_couples.append([picked, placed])
#         pick_place_combo[j] = pick_place_couples
#
#     return states


def tot_objects_placed(s):
    # Given the state s of the system, returns a list where each entry j specifies how many objects of type j
    # have already been placed
    placed = [0] * O
    for j in range(len(placed)):
        for t in range(T):
            placed[j] += s[3 + (T+1) * j + t]
    return placed


def objects_collected(s):
    # Yields the number of objects collected but not yet placed (inferred from the state entries)
    # INPUT
    # s: current state of the system
    # OUTPUT
    # obj: the number of objects collected but not yet placed in the given state
    objs = 0
    placed = tot_objects_placed(s)
    for j in range(O):
        objs += s[2 + (T+1) * j] - placed[j]  # picked - placed
    return objs


def state_space_brute_force(tot_obj_dict, maxObj):
    # Generates all possible final states in terms of picked and placed object types, without accounting for
    # time and node position. Tailored for specific missions.
    # INPUT
    # tot_obj_dict: {object: total amount to be placed during mission}, output of objects4mission_dict(mission)
    # maxObj: maximum number of objects to be carried at a time by the robot
    # OUTPUT
    # all_final_state: list of all possible states configurations

    # 1) Definition of all possible states such that picked[i]>=placed[i] for all objects i

    # # TEST INSTANCE
    # all_states = [[a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3, e1, e2, e3]
    #               for a1 in range(list(tot_obj_dict.values())[0] + 1)
    #               for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
    #               for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
    #               for b1 in range(list(tot_obj_dict.values())[1] + 1)
    #               for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
    #               for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)
    #               for c1 in range(list(tot_obj_dict.values())[2] + 1)
    #               for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)
    #               for c3 in range(0, list(list(mission.values())[1].values())[2] + 1)
    #               for d1 in range(list(tot_obj_dict.values())[3] + 1)
    #               for d2 in range(0, list(list(mission.values())[0].values())[3] + 1)
    #               for d3 in range(0, list(list(mission.values())[1].values())[3] + 1)
    #               for e1 in range(list(tot_obj_dict.values())[4] + 1)
    #               for e2 in range(0, list(list(mission.values())[0].values())[4] + 1)
    #               for e3 in range(0, list(list(mission.values())[1].values())[4] + 1)]

    # # MINI INSTANCE (3 obj types)
    # all_states = [[a1, a2, b1, b2, c1, c2]
    #               for a1 in range(list(tot_obj_dict.values())[0] + 1)
    #               for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
    #               for b1 in range(list(tot_obj_dict.values())[1] + 1)
    #               for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
    #               for c1 in range(list(tot_obj_dict.values())[2] + 1)
    #               for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)]

    # # MEDIUM INSTANCE
    # all_states = [[a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, d2, d3]
    #               for a1 in range(list(tot_obj_dict.values())[0] + 1)
    #               for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
    #               for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
    #               for b1 in range(list(tot_obj_dict.values())[1] + 1)
    #               for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
    #               for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)
    #               for c1 in range(list(tot_obj_dict.values())[2] + 1)
    #               for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)
    #               for c3 in range(0, list(list(mission.values())[1].values())[2] + 1)
    #               for d1 in range(list(tot_obj_dict.values())[3] + 1)
    #               for d2 in range(0, list(list(mission.values())[0].values())[3] + 1)
    #               for d3 in range(0, list(list(mission.values())[1].values())[3] + 1)]

    # LARGE INSTANCE and LARGE II
    all_states = [[a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4]
                  for a1 in range(list(tot_obj_dict.values())[0] + 1)
                  for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
                  for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
                  for a4 in range(0, list(list(mission.values())[2].values())[0] + 1)
                  for b1 in range(list(tot_obj_dict.values())[1] + 1)
                  for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
                  for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)
                  for b4 in range(0, list(list(mission.values())[2].values())[1] + 1)
                  for c1 in range(list(tot_obj_dict.values())[2] + 1)
                  for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)
                  for c3 in range(0, list(list(mission.values())[1].values())[2] + 1)
                  for c4 in range(0, list(list(mission.values())[2].values())[2] + 1)
                  for d1 in range(list(tot_obj_dict.values())[3] + 1)
                  for d2 in range(0, list(list(mission.values())[0].values())[3] + 1)
                  for d3 in range(0, list(list(mission.values())[1].values())[3] + 1)
                  for d4 in range(0, list(list(mission.values())[2].values())[3] + 1)
                  for e1 in range(list(tot_obj_dict.values())[4] + 1)
                  for e2 in range(0, list(list(mission.values())[0].values())[4] + 1)
                  for e3 in range(0, list(list(mission.values())[1].values())[4] + 1)
                  for e4 in range(0, list(list(mission.values())[2].values())[4] + 1)]

    # # INSTANCE FOR GRAPHS
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

    # # LARGE INSTANCE I
    # all_states = [[a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4]
    #               for a1 in range(list(tot_obj_dict.values())[0] + 1)
    #               for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
    #               for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
    #               for a4 in range(0, list(list(mission.values())[2].values())[0] + 1)
    #               for b1 in range(list(tot_obj_dict.values())[1] + 1)
    #               for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
    #               for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)
    #               for b4 in range(0, list(list(mission.values())[2].values())[1] + 1)
    #               for c1 in range(list(tot_obj_dict.values())[2] + 1)
    #               for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)
    #               for c3 in range(0, list(list(mission.values())[1].values())[2] + 1)
    #               for c4 in range(0, list(list(mission.values())[2].values())[2] + 1)]

    # 2) Filtering the states such that the robot does not carry more than maxObj objects
    all_states_b = []
    for s in all_states:
        carriedObj = objects_collected([0, 0] + s)
        if 0 <= carriedObj <= maxObj:
            all_states_b.append(s)

    # 3) Filtering the states such that picked >= tot_placed
    all_states_c = []
    for s in all_states_b:
        placed = tot_objects_placed([0, 0] + s)
        check = 0
        for j in range(O):
            if s[(T+1) * j] >= placed[j]:
                check += 1
        if check == O:
            all_states_c.append(s)

    return all_states_c


# ############################################# ACTION SPACE DEFINITION ############################################# #

# Moving is allowed from each node to all others. Info on time and risk in nodes_connections.
def moving_actions(s):
    # Returns all possible moving actions given the current state s
    # OUTPUT
    # move_actions: list containing all possible moving actions, identified by the destination node
    move_actions = []
    all_destinations = nodes.copy()
    all_destinations.remove(s[1])
    for k in all_destinations:
        if s[0] + nodes_connections[(s[1], k)]['time'] <= Thorizon:
            move_actions.append(k)

    if len(move_actions) == 0:
        # No moving action allowed: time horizon reached
        move_actions = None

    return move_actions


# Picking is only allowed in picking nodes, and the robot can only pick the object in the respective box
def picking_actions(s):
    # Returns the possible picking action given the current state s
    # OUTPUT
    # pick_action: picking action identified by the type of object to pick (A=0, B=1, C=2, etc.)
    pick_action = None
    if s[0] + tpick <= Thorizon:
        for j in range(len(objects_pick_nodes)):
            if list(objects_pick_nodes.values())[j] == s[1] and s[2 + (T+1) * j] < list(obj4mission_dict.values())[j] \
                    and objects_collected(s) < maxPortableObjs:
                pick_action = j

    return pick_action


# Placing is allowed only in placing nodes
def placing_actions(s):
    # Returns all possible placing actions given the current state s
    # OUTPUT
    # place_actions: list containing all possible placing actions, identified by the object type to place and the tray
    # where to place it (A=0, B=1, etc. and trays as tray0=0, tray1=1. etc.)
    place_actions = []
    if s[1] not in placing_nodes:
        # Robot is not in a placing location
        return None
    if s[0] + tplace > Thorizon:
        return None
    for t in range(T):
        for j in range(O):
            placed = tot_objects_placed(s)[j]
            # picked > placed & mission not yet completed & tray associated to placing node
            if s[2 + (T+1) * j] > placed and s[3 + (T+1) * j + t] < list(mission[trays[t]].values())[j] \
                    and s[1] == placing_nodes[t]:
                place_actions.append([j, t])

    if len(place_actions) == 0:
        # No placing action allowed: time horizon reached
        place_actions = None

    return place_actions


# ######################################## STATE TRANSITIONS DEFINITION ######################################## #

def new_state_moving(destination, s):
    # Returns the new state after moving action to destination, given current state s and exploiting information on
    # their connection {'time': , 'risk': , 'path': }
    new_s = s.copy()
    new_s[0] += nodes_connections[(s[1], destination)]['time']
    new_s[1] = destination
    return new_s


def new_state_picking(a, s):
    # Returns the new state after picking action a (= index of object type to pick : A=0, B=1, etc.),
    # given current state s
    new_s = s.copy()
    new_s[0] += tpick
    new_s[2 + (T+1) * a] += 1
    return new_s


def new_state_placing(a, s):
    # Returns the new state after placing action a (= [index of object type to place: A=0, B=1, etc., index of tray]),
    # given current state s
    new_s = s.copy()
    new_s[0] += tplace
    new_s[3 + (T+1) * a[0] + a[1]] += 1
    return new_s


# ######################################## ADMISSIBLE ACTIONS DEFINITION ######################################## #

def transition_matrix_Q(s):
    # Returns the transition matrix Q of the system in the current state s as a nested dictionary
    # The main key is the type of action: 'move', 'pick', 'place'
    # The secondary keys are the specific actions specifying where to move or what to pick and/or place and where
    # To each secondary key it is specified the state that is reached if the respective action is performed
    # Example: Q = {'move': {'npo': s'}, 'pick': {2: s''}, 'place': {[1, 0]: s'''}}

    Q_dict = {}

    if s[0] >= Thorizon:
        # Time horizon reached
        return None
    if tot_objects_placed(s) == list(obj4mission_dict.values()):
        # Mission accomplished
        return None

    move = moving_actions(s)
    pick = picking_actions(s)
    place = placing_actions(s)

    if move is None and pick is None and place is None:
        # No action can be performed: time horizon may be exceeded
        return None

    move_states = {}
    if move is not None:
        move = list(filter(lambda x: x is not None, move))
        for k in move:
            move_states[k] = new_state_moving(k, s)
        Q_dict['move'] = move_states

    pick_state = {}
    if pick is not None:
        pick_state[pick] = new_state_picking(pick, s)
        Q_dict['pick'] = pick_state

    place_states = {}
    if place is not None:
        place = list(filter(lambda x: x is not None, place))
        for k in place:
            place_states[tuple(k)] = new_state_placing(k, s)
        Q_dict['place'] = place_states

    return Q_dict


# ####################################### TERMINAL VALUE FUNCTION DEFINITION ####################################### #

# Terminal value function I
def terminal_value(s):
    # Value function for a terminal state s
    V = Thorizon - s[0]
    placed = tot_objects_placed(s)
    for j in range(O):
        V -= list(obj4mission_dict.values())[j] - placed[j]
        V += s[2 + (T+1) * j]
    return V

# # Terminal value function II
# def terminal_value(s):
#     # Value function for a terminal state s
#     V = 0.5 * (Thorizon - s[0])
#     placed = objects_placed(s)
#     for j in range(O):
#         V -= 15 * (list(obj4mission_dict.values())[j] - placed[j])
#     return V


# # Terminal value function III
# def terminal_value(s):
#     # Value function for a terminal state s
#     V = 5 * (Thorizon - s[0])
#     placed = tot_objects_placed(s)
#     for j in range(O):
#         V -= rewards['place'] * (list(obj4mission_dict.values())[j] - placed[j])
#         V += rewards['pick'] * s[2 + (T+1) * j]
#     return V


# ################################################# BACKWARD PASS ################################################# #

def backward_pass(discount_factor=0.99):
    V = {}  # Dictionary that maps each state to its value function
    for t in range(Thorizon, -1, -1):
        for n in nodes:
            for z in states_list:
                s = [t, n] + z
                Q = transition_matrix_Q(s)
                if Q is None:
                    V[tuple(s)] = terminal_value(s)
                else:
                    V_temp = []
                    for action_type in Q.keys():
                        if action_type == 'pick':
                            immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon
                            # or scale by 1 / (s[0] + 2)
                        elif action_type == 'place':
                            immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon
                            # or scale by 1 / (s[0] + 1)
                        else:
                            immediate_contribution = rewards[action_type]
                        for k in Q[action_type].values():
                            V_temp.append(immediate_contribution + discount_factor * V[tuple(k)])
                    V[tuple(s)] = max(V_temp)
    return V


# ################################################# FORWARD PASS ################################################# #

def forward_pass(initial_state, values):
    # Defines the optimal strategy to complete the mission starting from 'initial_state',
    # given the value functions for each state in 'values'
    # OUTPUT
    # action_sequence: sequence of optimal actions
    # states_sequence: optimal state sequence
    s = initial_state
    action_sequence = []
    state_sequence = [s]
    value_sequence = [values[tuple(s)]]
    Q = transition_matrix_Q(s)
    best_action = None
    best_new_state = None
    while Q is not None:  # Time horizon not reached or mission not completed yet
        max_value = -100000
        for action_type in Q.keys():
            if action_type == 'pick':
                immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon  # or * 1 / (s[0] + 2)
            elif action_type == 'place':
                immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon  # or * 1 / (s[0] + 1)
            else:
                immediate_contribution = rewards[action_type]
            for k in Q[action_type].keys():
                candidate_state = Q[action_type][k]
                candidate_value = immediate_contribution + values[tuple(candidate_state)]
                if candidate_value > max_value:
                    max_value = candidate_value
                    best_action = action_type + ' ' + str(k)
                    best_new_state = candidate_state
        action_sequence.append(best_action)
        state_sequence.append(best_new_state)
        value_sequence.append(max_value)
        s = best_new_state
        Q = transition_matrix_Q(s)

    return action_sequence, state_sequence, value_sequence


# ################################################# MAIN EXECTUTION ################################################# #

exec_time = [0] * 10
for it in range(10):

    start = timeit.default_timer()
    time.sleep(1)

    obj4mission_dict = objects4mission_dict(mission)

    states_list = state_space_brute_force(obj4mission_dict, maxPortableObjs)
    print(len(states_list))

    valueF = backward_pass(discount_factor=1)
    init = [0] * (O * (T + 1) + 2)
    # init[1] = random.choice(nodes)
    init[1] = 'np0'

    actions, states, value_seq = forward_pass(init, valueF)

    print(actions)
    print(states)
    print('Sequence of values associated to states: ' + str(value_seq))

    end = timeit.default_timer()

    exec_time[it] = end-start

    # print(f"Time taken is {end - start}s")

print(f"Mean execution time is {sum(exec_time)/10}")
