import math
import pickle as pkl
import os
import random
import timeit
import time


# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

nodes              = input_data["nodes"][:8]            # list of nodes where actions can be performed
# nodes              = ['np0', 'np1', 'np2', 'nt0', 'nt1', 'nt2']  # for Large instance I
# nodes              = ['np0', 'np1', 'np2', 'nt0']       # for MINI instance
# nodes = ['np0', 'np1', 'np2', 'nt0', 'nt1']  # for instance for Graphs
# nodes              = ['np0', 'np1', 'nt0', 'nt1']       # for instance for Graphs II
nodes_coordinates = input_data["nodes_coordinates"]  # dictionary n:c, where n-> node, c-> (x,y) coordinates
# take [:3] for MINI instance and instance for graphs, [:2] for instance for graphs II
objects = input_data["objects"]                       # list of objects that can be picked
O = len(objects)  # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]  # dictionary o:n, where o-> object, n-> picking node
# take [:1] for MINI instance
trays = input_data["trays"]  # list of target trays where objects can be place
trays.append('tray2')                                   # for LARGE instance, need a third tray
T = len(trays)  # total number of trays
trays_coordinates = input_data["trays_coordinates"]  # dictionary t:c, where t-> tray, c-> (x,y) coordinates
trays_coordinates['tray2'] = (131.75, 148.0)            # for LARGE instance, need a third tray
nodes_connections = input_data["nodes_connections"]  # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

Thorizon = 360  # (s) horizon of the optimization
throwing_nodes = nodes[5:8]  # for LARGE instance
# throwing_nodes = nodes[5:7]  # for TEST instance
# throwing_nodes = [nodes[-1]]  # for MINI instance
# throwing_nodes = nodes[3:5]  # for instance for Graph
# throwing_nodes = nodes[2:4]  # for instance for Graph II
# throwing_nodes = nodes[3:6]  # for instance for LARGE I

tpick = 7  # (s) time elapsed to perform picking action
tthrow = 5  # (s) time elapsed to perform throwing action
maxPortableObjs = 4
rewards = {'move': 0, 'pick': 10, 'throw': 12}
fail_rewards = {'move': -2, 'pick': 10, 'throw': 0}
rewards_adp_comp = {'move': 0, 'pick': 20, 'throw': 25}


# ############################################# MISSION DEFINITION ############################################# #

# # LARGE instance I (previously Large2)
# mission = {"tray0": {"objectA": 3, "objectB": 2, "objectC": 2},
#            "tray1": {"objectA": 0, "objectB": 2, "objectC": 4},
#            "tray2": {"objectA": 1, "objectB": 4, "objectC": 0}}

# LARGE instance II
mission = {"tray0": {"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 1, "objectE": 2},
           "tray1": {"objectA": 0, "objectB": 0, "objectC": 3, "objectD": 2, "objectE": 0},
           "tray2": {"objectA": 2, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 1}}

# # LARGE instance
# mission = {"tray0": {"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 0, "objectE": 1},
#            "tray1": {"objectA": 0, "objectB": 2, "objectC": 1, "objectD": 3, "objectE": 0},
#            "tray2": {"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 0, "objectE": 1}}

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
M = 15


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
        # v = 0
        return terminal_value(s)


def myopic_decision(s):
    # Calculates admissible actions given the state and returns the best myopic decision
    a = all_admissible_actions(s)  # Dictionary
    if len(a) == 0:
        return None, None, None
    C = 0
    best_action_type = 'move'
    for action_type in a.keys():
        if rewards[action_type] > C:
            best_action_type = action_type
            C = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon
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
        reward *= (Thorizon * 2 - s[0]) / Thorizon
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


# ####################################### TERMINAL VALUE FUNCTION DEFINITION ####################################### #

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


# # ################################################## FORWARD PASS ################################################## #
#
# start = timeit.default_timer()
# time.sleep(1)
#
# objective_values = []
# for iters in range(50):
#
#     initial_state = [0] * (O * (T + 1) + 2)
#     initial_state[1] = 'np0'
#
#
#     def forward_pass(initial_s):
#         # Defines the optimal strategy to complete the mission following a Myopic Rollout starting from 'initial_s',
#         # OUTPUT
#         # action_sequence: sequence of optimal actions
#         # states_sequence: optimal state sequence
#         s = initial_s
#         objective = 0
#         action_sequence = []
#         state_sequence = [s]
#         Q = transition_matrix_Q(s)
#         best_action = None
#         best_new_state = None
#         while Q is not None:  # Time horizon not reached or mission not completed yet
#             for action_type in Q.keys():
#                 max_V = -100
#                 for state in list(Q[action_type].values()):
#                     # For every admissible future state compute a rollout policy
#                     # V = Q(s, a) for a that brings the system to "state"
#                     V = rewards[action_type] * (Thorizon * 2 - s[0]) / Thorizon + myopic_rollout(state, -1)
#                     if V > max_V:
#                         max_V = V
#                         best_new_state = state.copy()
#                         best_action_type = action_type
#                         best_a = [a for a in Q[action_type] if Q[action_type][a] == best_new_state]
#                         best_a = best_a[0]
#                         best_action = best_action_type + ' ' + str(best_a)
#                     if best_action_type == 'throw':
#                         p_success = throwing_success(nodes_coordinates[s[1]], trays_coordinates[trays[best_a[1]]])
#                         if random.random() > p_success:
#                             best_new_state = failed_new_state_throwing(best_a, s)
#                             best_action = best_action + ' ' + '(FAILURE)'
#                     if best_action_type == 'move':
#                         p_risk = nodes_connections[(s[1], best_a)]['risk'] / 100
#                         if random.random() < p_risk:
#                             best_new_state = failed_new_state_moving(best_a, s)
#                             best_action = best_action + ' ' + '(COLLISION)'
#             state_sequence.append(best_new_state)
#             action_sequence.append(best_action)
#             s = best_new_state.copy()
#             Q = transition_matrix_Q(s)
#
#         objective += terminal_state_evaluation(s)
#
#         return action_sequence, state_sequence, objective
#
#     actions_seq, states_seq, objective_value = forward_pass(initial_state)
#
#     objective_values.append(objective_value)
#     print(actions_seq)
#     print(states_seq)
#     print('Terminal state evaluation: ' + str(objective_value))
#
#     end = timeit.default_timer()

#    print(f"Time taken is {end - start}s")

# print('Mean terminal state value: ' + str(sum(objective_values)/len(objective_values)))
