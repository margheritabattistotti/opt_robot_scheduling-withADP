import pickle as pkl
import os
import math
import random
import timeit
import time


# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

# nodes              = input_data["nodes"][:8]            # list of nodes where actions can be performed
# nodes              = input_data["nodes"][:7]            # list of nodes where actions can be performed
# nodes.remove('np4')                                     # for MEDIUM instance
nodes              = ['np0', 'np1', 'np2', 'nt0']       # for MINI instance to compare with DANI
# nodes              = ['np0', 'np1', 'np2', 'nt0', 'nt1']       # for instance for Graphs
# nodes              = ['np0', 'np1', 'nt0', 'nt1']       # for instance for Graphs II
nodes_coordinates  = input_data["nodes_coordinates"]    # dictionary n:c, where n-> node, c-> (x,y) coordinates
# take [:2] for instance for graph II, [:3] for MINI instance, [:4] for MEDIUM, all for LARGE
objects            = input_data["objects"][:3]              # list of objects that can be picked
O                  = len(objects)                       # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
trays              = input_data["trays"][:1]               # list of target trays where objects can be place
# trays.append('tray2')                                   # for LARGE instance, need a third tray
T                  = len(trays)                         # total number of trays
# T                  = 1                                  # for MINI instance
trays_coordinates  = input_data["trays_coordinates"]    # dictionary t:c, where t-> tray, c-> (x,y) coordinates
# trays_coordinates['tray2'] = (131.75, 148.0)            # for LARGE instance, need a third tray
nodes_connections  = input_data["nodes_connections"]    # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0

# ADDING RISK IN CONNECTIONS BETWEEN ADJACENT NODES TO CHECK IF IN SUCH CASE THE ALGORITHM PREFERS THE (NO-MORE)
# EQUIVALENT DIRECT PACT BETWEEN FAR APART NODES
# nodes_connections[('np0', 'np1')]['risk'] = 5
# nodes_connections[('np1', 'np0')]['risk'] = 5
# nodes_connections[('np1', 'np2')]['risk'] = 5
# nodes_connections[('np2', 'np1')]['risk'] = 5
# nodes_connections[('np2', 'np3')]['risk'] = 5
# nodes_connections[('np3', 'np2')]['risk'] = 5
# nodes_connections[('np3', 'np4')]['risk'] = 5
# nodes_connections[('np4', 'np3')]['risk'] = 5


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

# Thorizon = 150  # (s) horizon of the optimization
Thorizon = 120  # for MINI instance
# Thorizon = 200  # (s) for LARGE instance and instance for Graphs II
# Thorizon = 230  # for instance for Graphs
tpick = 7  # (s) time elapsed to perform picking action
tthrow = 5  # (s) time elapsed to perform throwing action


# ############################################# MISSION DEFINITION ############################################# #

# # TEST instance
# mission = {"tray0": {"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 0},
#            "tray1": {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 3, "objectE": 0}}

# MINI instance to compare with DANI
mission = {"tray0":  {"objectA": 3, "objectB": 2, "objectC": 2}}

# # MEDIUM instance
# mission = {"tray0": {"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0},
#            "tray1": {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 3}}

# # LARGE instance
# mission = {"tray0": {"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 0, "objectE": 1},
#            "tray1": {"objectA": 0, "objectB": 2, "objectC": 1, "objectD": 3, "objectE": 0},
#            "tray2": {"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 0, "objectE": 1}}

# # Instance for Graphs
# mission = {"tray0": {"objectA": 3, "objectB": 1, "objectC": 2},
#            "tray1": {"objectA": 2, "objectB": 2, "objectC": 1}}

# # Instance for Graphs II
# mission = {"tray0": {"objectA": 4, "objectB": 4},
#            "tray1": {"objectA": 4, "objectB": 4}}


# Defined by me
maxPortableObjs = 4
rewards = {'move': 0, 'pick': 10, 'throw': 12}
rewards_adp_comp = {'move': 0, 'pick': 20, 'throw': 25}
# throwing_nodes = nodes[5:7]  # for TEST instance
throwing_nodes = [nodes[-1]]  # for MINI instance to compare with DANI
# throwing_nodes = nodes[4:6]  # for MEDIUM instance
# throwing_nodes = nodes[5:8]  # for LARGE instance
# throwing_nodes = nodes[3:5]  # for instance for Graphs
# throwing_nodes = nodes[2:4]  # for instance for Graphs II


# ################################### DEFINITION OF THROWING SUCCESS PROBABILITY ################################### #

def throwing_success(c0, c1):
    # Computes the probability of throwing success given the coordinates of the starting and the destination nodes
    # INPUT
    # c0: coordinates of node from where throwing action is performed
    # c1: coordinates of destination (must be tray)
    # OUTPUT
    # p: probability of success
    dst = math.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2)
    if dst >= 80:
        p = 0
    else:
        p = -1 / 72 * (dst - 80)
    return p


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
    final_s = [0] * (2 + (T + 1) * num_obj)
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
    # final time and node position. Tailored for specific missions.
    # INPUT
    # tot_obj_dict: {object: total amount to be placed during mission}, output of objects4mission_dict(mission)
    # maxObj: maximum number of objects to be carried at a time by the robot
    # OUTPUT
    # all_final_state: list of all possible final state configurations

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

    # MINI INSTANCE (3 obj types)
    all_states = [[a1, a2, b1, b2, c1, c2]
                  for a1 in range(list(tot_obj_dict.values())[0] + 1)
                  for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
                  for b1 in range(list(tot_obj_dict.values())[1] + 1)
                  for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
                  for c1 in range(list(tot_obj_dict.values())[2] + 1)
                  for c2 in range(0, list(list(mission.values())[0].values())[2] + 1)]

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

    # # LARGE INSTANCE
    # all_states = [[a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4, e1, e2, e3, e4]
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
    #               for c4 in range(0, list(list(mission.values())[2].values())[2] + 1)
    #               for d1 in range(list(tot_obj_dict.values())[3] + 1)
    #               for d2 in range(0, list(list(mission.values())[0].values())[3] + 1)
    #               for d3 in range(0, list(list(mission.values())[1].values())[3] + 1)
    #               for d4 in range(0, list(list(mission.values())[2].values())[3] + 1)
    #               for e1 in range(list(tot_obj_dict.values())[4] + 1)
    #               for e2 in range(0, list(list(mission.values())[0].values())[4] + 1)
    #               for e3 in range(0, list(list(mission.values())[1].values())[4] + 1)
    #               for e4 in range(0, list(list(mission.values())[2].values())[4] + 1)]

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

    # # INSTANCE FOR GRAPHS II
    # all_states = [[a1, a2, a3, b1, b2, b3]
    #               for a1 in range(list(tot_obj_dict.values())[0] + 1)
    #               for a2 in range(0, list(list(mission.values())[0].values())[0] + 1)
    #               for a3 in range(0, list(list(mission.values())[1].values())[0] + 1)
    #               for b1 in range(list(tot_obj_dict.values())[1] + 1)
    #               for b2 in range(0, list(list(mission.values())[0].values())[1] + 1)
    #               for b3 in range(0, list(list(mission.values())[1].values())[1] + 1)]

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
    # mov_actions: list containing all possible moving actions, identified by the destination node
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


# Throwing is allowed only in placing nodes.
def throwing_actions(s):
    # Returns all possible throwing actions given the current state s
    # OUTPUT
    # throw_actions: list containing all possible throwing actions, identified by the object type to place and the tray
    # where to throw it (A=0, B=1, etc. and trays as tray0=0, tray1=1. etc.)
    throw_actions = []
    if s[1] not in throwing_nodes:
        # Robot is not in a throwing location
        return None
    if s[0] + tthrow > Thorizon:
        return None
    for t in range(T):
        for j in range(O):
            placed = tot_objects_placed(s)[j]
            # picked > placed & mission not yet completed
            if s[2 + (T+1) * j] > placed and s[3 + (T+1) * j + t] < list(mission[trays[t]].values())[j]:
                throw_actions.append([j, t])

    if len(throw_actions) == 0:
        # No throwing action allowed: time horizon reached
        throw_actions = None

    return throw_actions


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


def new_state_throwing(a, s):
    # Returns the new state after throwing action a (= [index of object type to place: A=0, B=1, etc., index of tray]),
    # given current state s
    new_s = s.copy()
    new_s[0] += tthrow
    new_s[3 + (T+1) * a[0] + a[1]] += 1
    return new_s


# ######################################## ADMISSIBLE ACTIONS DEFINITION ######################################## #

def transition_matrix_Q(s):
    # Returns the transition matrix Q of the system in the current state s as a nested dictionary
    # The main key is the type of action: 'move', 'pick', 'throw'
    # The secondary keys are the specific actions specifying where to move or what to pick and/or throw and where
    # To each secondary key it is specified the state that is reached if the respective action is performed
    # Example: Q = {'move': {'npo': s'}, 'pick': {2: s''}, 'throw': {[1, 0]: s'''}}

    Q_dict = {}

    if s[0] >= Thorizon:
        # Time horizon reached
        return None
    if tot_objects_placed(s) == list(obj4mission_dict.values()):
        # Mission accomplished
        return None

    move = moving_actions(s)
    pick = picking_actions(s)
    throw = throwing_actions(s)

    if move is None and pick is None and throw is None:
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

    throw_states = {}
    if throw is not None:
        throw = list(filter(lambda x: x is not None, throw))
        for k in throw:
            throw_states[tuple(k)] = new_state_throwing(k, s)
        Q_dict['throw'] = throw_states

    return Q_dict


# ####################################### TERMINAL VALUE FUNCTION DEFINITION ####################################### #

# # Terminal value function I
# def terminal_value(s):
#     # Value function for a terminal state s
#     V = Thorizon - s[0]
#     placed = tot_objects_placed(s)
#     for j in range(O):
#         V -= list(obj4mission_dict.values())[j] - placed[j]
#         V += s[2 + (T+1) * j]
#     return V

# # Terminal value function II
# def terminal_value(s):
#     # Value function for a terminal state s
#     V = 0.5 * (Thorizon - s[0])
#     placed = tot_objects_placed(s)
#     for j in range(O):
#         V -= 15 * (list(obj4mission_dict.values())[j] - placed[j])
#     return V

# Terminal value function III
def terminal_value(s):
    # Value function for a terminal state s
    V = 5 * (Thorizon - s[0])
    placed = tot_objects_placed(s)
    for j in range(O):
        V -= rewards['throw'] * (list(obj4mission_dict.values())[j] - placed[j])
        V += rewards['pick'] * s[2 + (T+1) * j]
    return V


def terminal_state_evaluation(s):
    # Value function for a terminal state s
    V = 5 * (Thorizon - s[0])
    placed = tot_objects_placed(s)
    for j in range(O):
        V -= rewards_adp_comp['throw'] * (list(obj4mission_dict.values())[j] - placed[j])
        V += rewards_adp_comp['pick'] * s[2 + (T+1) * j]
    return V


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
                        elif action_type == 'throw':
                            immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon
                            # or scale by 1 / (s[0] + 2)
                        else:
                            immediate_contribution = rewards[action_type]
                        for k in Q[action_type].keys():
                            v = immediate_contribution + discount_factor * V[tuple(Q[action_type][k])]
                            if action_type == 'throw':
                                p_success = throwing_success(nodes_coordinates[s[1]], trays_coordinates[trays[k[1]]])
                                failed_s_t = s.copy()
                                failed_s_t[0] += tthrow
                                failed_s_t[2 + (T+1) * k[0]] -= 1  # Throwing failed: object lost
                                v = v * p_success + (1 - p_success) * (discount_factor * V[tuple(failed_s_t)])
                            elif action_type == 'move':
                                # NB risk is a percentage
                                p_risk = nodes_connections[(s[1], Q[action_type][k][1])]['risk'] / 100
                                failed_s_m = Q[action_type][k].copy()
                                if failed_s_m[0] > Thorizon - 5:
                                    failed_s_m[0] = Thorizon
                                else:
                                    failed_s_m[0] += 5  # Reach next state but with 5 seconds penalty
                                v = v * (1 - p_risk) + p_risk * (-2 + discount_factor * V[tuple(failed_s_m)])
                            V_temp.append(v)

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
    objective = 0
    action_sequence = []
    state_sequence = [s]
    Q = transition_matrix_Q(s)
    best_action = None
    best_new_state = None
    while Q is not None:  # Time horizon not reached or mission not completed yet
        max_value = -100000
        for action_type in Q.keys():
            if action_type == 'pick':
                immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon  # or * 1 / (s[0] + 2)
            for k in Q[action_type].keys():
                candidate_state = Q[action_type][k]
                if action_type == 'move':
                    p_risk = nodes_connections[(s[1], Q[action_type][k][1])]['risk'] / 100
                    immediate_contribution = rewards[action_type] * (1 - p_risk) + p_risk * -2
                if action_type == 'throw':
                    p_success = throwing_success(nodes_coordinates[s[1]], trays_coordinates[trays[k[1]]])
                    immediate_contribution = p_success * rewards[action_type] * \
                        (Thorizon * 2 - s[0]) / Thorizon  # or * 1 / (s[0] + 2)
                candidate_value = immediate_contribution + values[tuple(candidate_state)]
                if candidate_value > max_value:
                    max_value = candidate_value
                    best_new_state = candidate_state
                    best_action = action_type + ' ' + str(k)
                    if action_type == 'throw' and \
                            random.random() > p_success:
                        best_new_state = s.copy()
                        best_new_state[0] += tthrow
                        best_new_state[2 + (T+1) * k[0]] -= 1  # Throwing failed: object lost
                        best_action = best_action + ' ' + '(FAILED)'
                    elif action_type == 'move' and \
                            random.random() < p_risk:
                        if best_new_state[0] > Thorizon - 5:
                            best_new_state[0] = Thorizon
                        else:
                            best_new_state[0] += 5  # Reach next state but with 5 second penalty
                        best_action = best_action + ' ' + '(COLLISION)'
        action_sequence.append(best_action)
        state_sequence.append(best_new_state)
        s = best_new_state
        Q = transition_matrix_Q(s)

    objective += terminal_state_evaluation(s)

    return action_sequence, state_sequence, objective


# ################################################# MAIN EXECTUTION ################################################# #

exec_time = [0] * 10
objective = [0] * 10
for it in range(10):

    start = timeit.default_timer()
    time.sleep(1)

    obj4mission_dict = objects4mission_dict(mission)

    states_list = state_space_brute_force(obj4mission_dict, maxPortableObjs)
    print(len(states_list))

    valueF = backward_pass(discount_factor=1)
    # with open('valueF_data_TEST_TF3.pkl', 'wb') as fp:
    #     pkl.dump(valueF, fp)
    # valueF = pkl.load(open(dir_path + "//valueF_data_mission2_TF3.pkl", "rb"))

    init = [0] * (O * (T + 1) + 2)
    # init[1] = random.choice(nodes)
    init[1] = 'np0'

    actions, states, objective_value = forward_pass(init, valueF)
    objective[it] = objective_value

    print(actions)
    print(states)
    print('Terminal state evaluation: ' + str(objective_value))

    end = timeit.default_timer()

    exec_time[it] = end-start

#  print(f"Time taken is {end - start}s")

print(f"Mean execution time is {sum(exec_time)/10}")
print('Mean terminal state value: ' + str(sum(objective)/len(objective)))
