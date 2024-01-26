import math
import pickle as pkl
import os
import random
import timeit
import time

# 23/12/2023 -> NON STA FUNZIONANDO:
# 1) riempie i vassoi in maniera casuale
# 2) si muove a casissimo (ma alzando i rewards si risolve forse)

# 29/12/2023 -> FUNZIONA ma:
# 1) pare peggio degli altri
# 2) se i tempi di consegna sono ben distanziati capisce meglio la priorità vedi medium instance - orders IV
# 3) large instance - in generale mantiene abbastanza le priorità ma non sempre, però non rispetta quasi mai
# i tempi di consegna

# TUTTO SOMMATO: C'è prospettiva di miglioramento, ma al momento non c'è tempo.
# Ci accontentiamo della strada giusta senza arrivare a destinazione? SVILUPPI FUTURI della tesi
# PRO: è pur sempre più veloce del sequential filling con DP.

# IN ASSOLUTO VINCE SEQUENTIAL TRAY FILLING HEURISTIC! VELOCE E PRECISO ANCHE PER ISTANZE MOLTO GRANDI (1 sec)

# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

nodes              = input_data["nodes"][:7]            # list of nodes where actions can be performed
nodes_coordinates  = input_data["nodes_coordinates"]    # dictionary n:c, where n-> node, c-> (x,y) coordinates
objects            = input_data["objects"]              # list of objects that can be picked
O                  = len(objects)                       # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
trays              = input_data["trays"]                # list of target trays where objects can be place
T                  = len(trays)                         # total number of trays
trays_coordinates  = input_data["trays_coordinates"]    # dictionary t:c, where t-> tray, c-> (x,y) coordinates
nodes_connections  = input_data["nodes_connections"]    # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

Thorizon = 600  # (s) horizon of the optimization
tpick = 7  # (s) time elapsed to perform picking action
tthrow = 5  # (s) time elapsed to perform throwing action


# ############################################# MISSION DEFINITION ############################################# #

def set_mission():

    # MEDIUM instance
    # ORDERS I, tot obj 4 + 6 + 3 = 13
    # global orders
    # orders = [[{"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 0, "objectE": 1}, 100],
    #           [{"objectA": 0, "objectB": 2, "objectC": 1, "objectD": 3, "objectE": 0}, 150],
    #           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 0, "objectE": 1}, 70]]
    # global rg_move_stream
    # rg_move_stream = random.Random(2404)
    # global rg_throw_stream
    # rg_throw_stream = random.Random(610)

    # # ORDERS II, tot obj 4 + 6 + 3 = 13
    # global orders
    # orders = [[{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 1}, 140],
    #           [{"objectA": 3, "objectB": 1, "objectC": 0, "objectD": 2, "objectE": 0}, 210],
    #           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 0, "objectE": 1}, 90]]
    # # rg_move_stream = random.Random(2810)
    # # rg_throw_stream = random.Random(310)

    # # ORDERS III, tot obj 5 + 5 + 4 = 14
    # global orders
    # orders = [[{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 2}, 130],
    #           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 3, "objectE": 0}, 190],
    #           [{"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 1}, 120]]
    # # rg_move_stream = random.Random(1505)
    # # rg_throw_stream = random.Random(191)

    # # ORDERS IV, tot obj 5 + 5 + 4 = 14
    # global orders
    # orders = [[{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 2}, 210],
    #           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 3, "objectE": 0}, 195],
    #           [{"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 1}, 180]]
    # # rg_move_stream = random.Random(2008)
    # # rg_throw_stream = random.Random(503)

    # ORDERS V, tot obj 5 + 4 + 3 = 12
    # Estimated picking and placing times:
    # i)   :     5x7 + 5x5 = 60
    # ii)  :     4x7 + 4x5 = 48
    # iii) :     3x7 + 3x5 = 36
    # global orders
    # orders = [[{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 2}, 210],
    #           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 2, "objectE": 0}, 90],
    #           [{"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 0}, 120]]
    # rg_move_stream = random.Random(1609)
    # rg_throw_stream = random.Random(793)

    # LARGE instance:
    # ORDERS I, tot obj 9 + 11 + 11 = 31
    global orders
    orders = [[{"objectA": 3, "objectB": 2, "objectC": 1, "objectD": 2, "objectE": 1}, 370],  # 108
              [{"objectA": 2, "objectB": 2, "objectC": 4, "objectD": 3, "objectE": 0}, 550],  # 132
              [{"objectA": 1, "objectB": 2, "objectC": 3, "objectD": 2, "objectE": 3}, 230]]   # 132
    global rg_move_stream
    rg_move_stream = random.Random(2404)
    global rg_throw_stream
    rg_throw_stream = random.Random(610)

    # Sorting the orders in ascending order of priority (highest priority = lowest delivery time)
    orders.sort(key=lambda x: x[1])

    global mission
    mission = {}
    global obj4trays_dict
    obj4trays_dict = {}
    global trays_delivery_times
    trays_delivery_times = []
    global trays_entering_times
    trays_entering_times = []
    global trays_completion_times
    trays_completion_times = []
    global orders_completion
    orders_completion = []
    # Suppose number of orders to be always greater than the number of available trays
    for tray in trays:
        mission[tray] = orders[0][0]
        obj4trays_dict[tray] = sum(list(mission[tray].values()))
        trays_delivery_times.append(orders[0][1])
        trays_entering_times.append(0)
        trays_completion_times.append(0)
        orders.remove(orders[0])


set_mission()


# Defined by me
maxPortableObjs = 4
rewards = {'move': 0, 'pick': 10, 'throw': 12}
fail_rewards = {'move': -2, 'pick': 10, 'throw': 0}
rewards_adp_comp = {'move': 0, 'pick': 20, 'throw': 25}
throwing_nodes = nodes[5:7]  # for MEDIUM instance


def tot_objects4mission():
    # Returns a list where each entry j specifies how many objects of type j must be placed during mission
    tot_object = [0] * O
    for j in range(O):
        for t in range(T):
            tot_object[j] += list(mission.values())[t][objects[j]]

    return tot_object


tot_obj4mission = tot_objects4mission()


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


# ############################################### ROLLOUT DEFINITION ############################################### #

gamma = 0.95
M = 10


def myopic_rollout(s, m=-1):
    # Recursively executes a rollout from state s, where m is the steps in the future counter
    best_action_type, best_action, best_contribution = myopic_decision(s)  # Call myopic decision
    if best_contribution is not None:
        m += 1
        if m < M:
            next_s, real_contribution, _ = simulation(s, best_action_type, best_action)
            if is_terminal(next_s) is False:
                V = myopic_rollout(next_s, m)
                v = real_contribution + gamma * V
            else:
                v = terminal_value(next_s)
        else:
            v = best_contribution
        return v  # Value of the state
    else:
        return terminal_value(s)


def myopic_decision(s):
    # Calculates admissible actions given the state and returns the best myopic decision
    a = all_admissible_actions(s)  # Dictionary
    if len(a) == 0:
        return None, None, None
    C = -100
    # Initialization of best action to an arbitrary action
    best_action_type = 'move'
    best_a = random.choice(list(a[best_action_type]))
    for action_type in a.keys():
        if action_type == 'throw':
            for k in a[action_type]:
                # No indicator function
                immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / trays_delivery_times[k[1]]
                # # Indicator function
                # immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / trays_delivery_times[k[1]] \
                #                          * (trays_delivery_times[k[1]] <= s[0]) + \
                #                          rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon * (
                #                                      trays_delivery_times[k[1]] > s[0])
                if immediate_contribution > C:
                    C = immediate_contribution
                    best_action_type = action_type
                    best_a = k
        else:
            immediate_contribution = rewards[action_type]
            if immediate_contribution > C:
                best_action_type = action_type
                C = immediate_contribution
                best_a = random.choice(list(a[best_action_type]))

    return best_action_type, best_a, C


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
        # No moving actions allowed: time horizon reached
        move_actions = None

    return move_actions


# Picking is only allowed in picking nodes, and the robot can only pick the object in the respective box
def picking_actions(s):
    # Returns the possible picking action given the current state s
    # OUTPUT
    # pick_action: picking action identified by the type of object to pick (A=0, B=1, C=2, etc.)
    pick_action = []
    if s[0] + tpick <= Thorizon:
        for j in range(O):
            if list(objects_pick_nodes.values())[j] == s[1] \
                    and s[2 + (T + 1) * j] < tot_obj4mission[j] \
                    and objects_collected(s) < maxPortableObjs:
                pick_action.append(j)

    if len(pick_action) == 0:
        pick_action = None

    return pick_action


# Throwing is allowed only in placing nodes
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
        # No time to perform action
        return None
    for t in range(T):
        for j in range(O):
            placed = tot_objects_placed(s)[j]
            # picked > placed & mission not yet completed
            if s[2 + (T + 1) * j] > placed \
                    and s[3 + (T + 1) * j + t] < list(mission[trays[t]].values())[j]:
                throw_actions.append([j, t])

    if len(throw_actions) == 0:
        # No throwing actions allowed: time horizon reached
        throw_actions = None

    return throw_actions


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
    if tot_objects_placed(s) == tot_obj4mission:
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
        p = pick[0]
        pick_state[p] = new_state_picking(p, s)
        Q_dict['pick'] = pick_state

    throw_states = {}
    if throw is not None:
        throw = list(filter(lambda x: x is not None, throw))
        for k in throw:
            throw_states[tuple(k)] = new_state_throwing(k, s)
        Q_dict['throw'] = throw_states

    return Q_dict


def all_admissible_actions(s):
    # Returns all admissible actions given the current state
    actions = {}

    move = moving_actions(s)
    pick = picking_actions(s)
    throw = throwing_actions(s)

    if move is not None:
        actions['move'] = move

    if pick is not None:
        actions['pick'] = pick

    if throw is not None:
        actions['throw'] = throw

    return actions


# ######################################## STATE TRANSITIONS DEFINITION ######################################## #

def new_state_moving(a, s):
    # Returns the new state after moving action to destination a, given current state s and exploiting information
    # on their connection {'time': , 'risk': , 'path': }
    next_s = s.copy()
    next_s[0] += nodes_connections[(s[1], a)]['time']
    next_s[1] = a

    return next_s


def failed_new_state_moving(a, s):
    # In case of collision, returns the new state after moving action to destination a, given current state s and
    # exploiting information on their connection {'time': , 'risk': , 'path': }
    next_s = s.copy()
    next_s[0] += nodes_connections[(s[1], a)]['time']
    if next_s[0] + 5 <= Thorizon:
        next_s[0] += 5
    else:
        next_s[0] = Thorizon
    next_s[1] = a

    return next_s


def new_state_picking(a, s):
    # Returns the new state after picking action a (= index of object type to pick : A=0, B=1, etc.),
    # given current state s
    next_s = s.copy()
    next_s[0] += tpick
    next_s[2 + (T + 1) * a] += 1

    return next_s


def new_state_throwing(a, s):
    # Returns the new state after throwing action a (= [index of object type to place: A=0, B=1, etc., index of tray]),
    # given current state s
    next_s = s.copy()
    next_s[0] += tthrow
    next_s[3 + (T + 1) * a[0] + a[1]] += 1

    return next_s


def failed_new_state_throwing(a, s):
    # In case of failure, returns the new state after throwing action a = [index of object to throw, index of tray
    # where to throw it], given current state s
    next_s = s.copy()
    next_s[0] += tthrow
    next_s[2 + (T + 1) * a[0]] -= 1

    return next_s


# ########################################### SIMULATION STEP DEFINITION ########################################### #

def simulation(s, action_type, a):
    # Give an action, simulates the future state of the system give the current one (s)
    reward = rewards[action_type]
    success = 1
    if action_type == 'move':
        p_risk = nodes_connections[(s[1], a)]['risk'] / 100
        if random.random() < p_risk:
            success = 0
            next_state = failed_new_state_moving(a, s)
            reward = fail_rewards['move']
        else:
            next_state = new_state_moving(a, s)
    elif action_type == 'throw':
        # No indicator function
        reward = rewards[action_type] * (Thorizon * 2 - s[0]) / trays_delivery_times[a[1]]
        # # Indicator function
        # reward = rewards[action_type] * (Thorizon * 2 - s[0]) / trays_delivery_times[a[1]] * (
        #         trays_delivery_times[a[1]] <= s[0]) + \
        #          rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon * (trays_delivery_times[a[1]] > s[0])
        p_success = throwing_success(nodes_coordinates[s[1]],
                                     trays_coordinates[trays[a[1]]])
        if random.random() > p_success:
            success = 0
            next_state = failed_new_state_throwing(a, s)
            reward = fail_rewards['throw']
        else:
            next_state = new_state_throwing(a, s)
    elif action_type == 'pick':
        reward *= (Thorizon * 2 - s[0]) / Thorizon
        next_state = new_state_picking(a, s)

    return next_state, reward, success


# ########################################### ADDITIONAL NEEDED FUNCTIONS ########################################### #

def is_terminal(s):
    # Checks if the current state s is a terminal state
    terminal = False
    # s is terminal if time horizon is reached or mission is completed
    if s[0] >= Thorizon or \
            tot_objects_placed(s) == tot_obj4mission or \
            len(all_admissible_actions(s)) == 0:
        terminal = True

    return terminal


def tot_objects_placed(s):
    # Given the state s of the system, returns a list where each entry j specifies how many objects of type j
    # have already been placed
    placed = [0] * O
    for j in range(O):
        for t in range(T):
            placed[j] += s[3 + (T + 1) * j + t]

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
        objs += s[2 + (T + 1) * j] - placed[j]  # picked - placed

    return objs


def trays_level(s):
    # Given the state s of the system, returns a list where each entry t specifies how many objects
    # have already been placed in tray t
    placed = [0] * T
    for t in range(T):
        for j in range(O):
            placed[t] += s[3 + (T+1) * j + t]
    return placed

# ####################################### TERMINAL VALUE FUNCTION DEFINITION ####################################### #

# # Terminal value function II
# def terminal_value(s):
#     # Value function for a terminal state s
#     V = 0.5 * (Thorizon - s[0])
#     placed = tot_objects_placed(s)
#     for j in range(O):
#         V -= 15 * (tot_obj4mission[j] - placed[j])
#     return V


# Terminal value function III -> Evaluation metric
def terminal_state_evaluation(s):
    # Value function for a terminal state s
    V = 5 * (Thorizon - s[0])
    placed = tot_objects_placed(s)
    for j in range(O):
        V -= rewards_adp_comp['throw'] * (tot_obj4mission[j] - placed[j])
        V += rewards_adp_comp['pick'] * s[2 + (T + 1) * j]
    return V


# Terminal value function IV
def terminal_value(s):
    # Value function for a terminal state s
    V = 0.5 * (Thorizon - s[0])
    placed = tot_objects_placed(s)
    for j in range(O):
        V -= tot_obj4mission[j] - placed[j]
        V += s[2 + (T + 1) * j]
    return V


# NOT THE BRIGHTEST IDEA
# ################# A state is terminal if one tray is full. In fact, must change tray -> new system ################ #

# def is_terminal(s):
#     # Checks if the current state s is a terminal state
#     terminal = False
#     if len(orders) == 0:
#         if s[0] >= Thorizon or \
#                 tot_objects_placed(s) == tot_obj4mission or \
#                 len(all_admissible_actions(s)) == 0:
#             terminal = True
#     else:
#         full_tray = 0
#         for t in range(T):
#             if trays_level(s)[t] == obj4trays_dict[trays[t]]:
#                 full_tray += 1
#         if s[0] >= Thorizon or \
#                 full_tray != 0 or \
#                 len(all_admissible_actions(s)) == 0:
#             terminal = True
#
#     return terminal
#
#
# # Terminal value function V
# def terminal_value(s):
#     # Value function for a terminal state s
#     if len(orders) == 0:
#         V = 0.5 * (Thorizon - s[0])
#         placed = tot_objects_placed(s)
#         for j in range(O):
#             V -= tot_obj4mission[j] - placed[j]
#             V += s[2 + (T + 1) * j]
#         return V
#
#     else:
#         V = 0
#         for t in range(T):
#             if trays_level(s)[t] == obj4trays_dict[trays[t]]:
#                 V = 10 * (trays_delivery_times[t] - s[0]) / trays_delivery_times[t]
#             else:
#                 for j in range(O):
#                     V += s[2 + (T + 1) * j + t]
#         return V


# ################################################## FORWARD PASS ################################################## #

def forward_pass(initial_s):
    # Defines the optimal strategy to complete the mission following a Myopic Rollout starting from 'initial_s',
    # OUTPUT
    # action_sequence: sequence of optimal actions
    # states_sequence: optimal state sequence
    s = initial_s
    objective = 0
    action_sequence = []
    state_sequence = [s]
    Q = transition_matrix_Q(s)
    best_action = None
    best_new_state = None
    while Q is not None:  # Time horizon not reached or mission not completed yet
        for action_type in Q.keys():
            max_V = -100
            for state in list(Q[action_type].values()):
                # For every admissible future state compute a rollout policy
                # V = Q(s, a) for a that brings the system to "state"
                if action_type == 'throw':
                    k = [k for k in Q[action_type] if Q[action_type][k] == state]
                    k = k[0]
                    # No indicator function
                    immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / trays_delivery_times[k[1]]
                    # # Indicator function
                    # immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0])/trays_delivery_times[k[1]] \
                    #                          * (trays_delivery_times[k[1]] <= s[0]) + \
                    #                          rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon * (
                    #                                  trays_delivery_times[k[1]] > s[0])
                else:
                    immediate_contribution = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon
                V = immediate_contribution + myopic_rollout(state, -1)
                if V > max_V:
                    max_V = V
                    best_new_state = state.copy()
                    best_action_type = action_type
                    best_a = [a for a in Q[action_type] if Q[action_type][a] == best_new_state]
                    best_a = best_a[0]
                    best_action = best_action_type + ' ' + str(best_a)
                if best_action_type == 'throw':
                    p_success = throwing_success(nodes_coordinates[s[1]], trays_coordinates[trays[best_a[1]]])
                    if rg_throw_stream.random() > p_success:
                        best_new_state = failed_new_state_throwing(best_a, s)
                        best_action = best_action + ' ' + '(FAILURE)'
                if best_action_type == 'move':
                    p_risk = nodes_connections[(s[1], best_a)]['risk'] / 100
                    if rg_move_stream.random() < p_risk:
                        best_new_state = failed_new_state_moving(best_a, s)
                        best_action = best_action + ' ' + '(COLLISION)'
        state_sequence.append(best_new_state)
        action_sequence.append(best_action)
        s = best_new_state.copy()

        check_full_tray = trays_level(s)
        # If a tray is full and another order is in line, change the mission by emptying such tray and assigning to
        # it the new order
        for t in range(T):
            if check_full_tray[t] == obj4trays_dict[trays[t]]:
                if trays_completion_times[t] == 0:
                    action_sequence.append(trays[t] + ' ' + 'is full!')
                    trays_completion_times[t] = s[0]
                    orders_completion.append({'Asked': trays_delivery_times[t], 'Actual': trays_completion_times[t]})
                if len(orders) != 0:
                    action_sequence.append('Must change it.')
                    trays_completion_times[t] = 0
                    # Updating mission and associated information
                    mission[trays[t]] = orders[0][0]
                    trays_delivery_times[t] = orders[0][1]
                    trays_entering_times[t] = s[0]
                    obj4trays_dict[trays[t]] = sum(list(mission[trays[t]].values()))
                    orders.remove(orders[0])
                    action_sequence.append('Now' + ' ' + trays[t] + ' ' + 'contains a new order'
                                           + ' ' + str(mission[trays[t]]))
                    global tot_obj4mission
                    tot_obj4mission = tot_objects4mission()
                    updated_s = s.copy()
                    # Updating current state
                    for j in range(O):
                        # Removing placed objects in tray t from picked entry
                        updated_s[2 + (T + 1) * j] -= updated_s[3 + (T + 1) * j + t]
                        # Resetting to zero the number of placed objects in tray t
                        updated_s[3 + (T + 1) * j + t] = 0
                    state_sequence.append('Changing ' + trays[t])
                    state_sequence.append(updated_s)
                    state_sequence.append(trays[t] + ' changed')
                    s = updated_s

        Q = transition_matrix_Q(s)

    objective += terminal_state_evaluation(s)

    return action_sequence, state_sequence, objective


# ################################################# MAIN EXECTUTION ################################################# #


start = timeit.default_timer()
time.sleep(1)

objective_values = []
for iters in range(50):

    initial_state = [0] * (O * (T + 1) + 2)
    initial_state[1] = 'np0'

    actions_seq, states_seq, objective_value = forward_pass(initial_state)

    objective_values.append(objective_value)
    print(actions_seq)
    print(states_seq)
    print('Terminal state evaluation: ' + str(objective_value))
    print('Delivery times:' + ' ' + str(orders_completion))

    end = timeit.default_timer()

    print(f"Time taken is {end - start}s")

    set_mission()
    tot_obj4mission = tot_objects4mission()

print('Mean terminal state value: ' + str(sum(objective_values)/len(objective_values)))
