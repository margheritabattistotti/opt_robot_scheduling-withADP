import pickle as pkl
import os
import random
import timeit
import time
import math


# ############################################# INPUT DATA EXTRACTION ############################################# #

dir_path = os.path.dirname(os.path.realpath(__file__))
input_data = pkl.load(open(dir_path + "//input_data//input_dict.p", "rb"))

nodes              = input_data["nodes"][:6]            # list of nodes where actions can be performed
nodes_coordinates  = input_data["nodes_coordinates"]    # dictionary n:c, where n-> node, c-> (x,y) coordinates
objects            = input_data["objects"]              # list of objects that can be picked
O                  = len(objects)                       # total number of object types
objects_pick_nodes = input_data["objects_pick_nodes"]   # dictionary o:n, where o-> object, n-> picking node
trays              = input_data["trays"]                # list of target trays where objects can be place
T                  = 1                                  # total number of trays
trays_coordinates  = input_data["trays_coordinates"]    # dictionary t:c, where t-> tray, c-> (x,y) coordinates
nodes_connections  = input_data["nodes_connections"]    # dictionary (n0,n1):(t,r,p), where (n0,n1)-> tuple of nodes,
# t-> time to reach n1 from n0, r-> navigation risk to reach n1 from n0, p-> path to reach n1 from n0


# ######################################## ADDITIONAL PARAMETERS DEFINITION ######################################## #

Thorizon = 600  # (s) horizon of the optimization
tpick = 7  # (s) time elapsed to perform picking action
tthrow = 5  # (s) time elapsed to perform throwing action


# ############################################# MISSION DEFINITION ############################################# #

# MEDIUM instance
# # ORDERS I, tot obj 4 + 6 + 3 = 13
# orders = [[{"objectA": 1, "objectB": 2, "objectC": 0, "objectD": 0, "objectE": 1}, 100],
#           [{"objectA": 0, "objectB": 2, "objectC": 1, "objectD": 3, "objectE": 0}, 150],
#           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 0, "objectE": 1}, 70]]
# # Two random number generator streams are fixed so that comparisons with other stochastic methods are more accurate
# rg_move_stream = random.Random(2404)
# rg_throw_stream = random.Random(610)

# # ORDERS II, tot obj 4 + 6 + 3 = 13
# orders = [[{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 1}, 140],
#           [{"objectA": 3, "objectB": 1, "objectC": 0, "objectD": 2, "objectE": 0}, 210],
#           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 0, "objectE": 1}, 90]]
# # Two random number generator streams are fixed so that comparisons with other stochastic methods are more accurate
# rg_move_stream = random.Random(2810)
# rg_throw_stream = random.Random(310)

# # ORDERS III, tot obj 5 + 5 + 4 = 14
# orders = [[{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 2}, 130],
#           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 3, "objectE": 0}, 190],
#           [{"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 1}, 120]]
# # Two random number generator streams are fixed so that comparisons with other stochastic methods are more accurate
# rg_move_stream = random.Random(1505)
# rg_throw_stream = random.Random(191)

# # ORDERS IV, tot obj 5 + 5 + 4 = 14
# orders = [[{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 2}, 210],
#           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 3, "objectE": 0}, 195],
#           [{"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 1}, 180]]
# # Two random number generator streams are fixed so that comparisons with other stochastic methods are more accurate
# rg_move_stream = random.Random(2008)
# rg_throw_stream = random.Random(503)

# # ORDERS V, tot obj 5 + 4 + 3 = 12
# # Estimated picking and placing times:
# # i)   :     5x7 + 5x5 = 60
# # ii)  :     4x7 + 4x5 = 48
# # iii) :     3x7 + 3x5 = 36
# orders = [[{"objectA": 1, "objectB": 1, "objectC": 1, "objectD": 0, "objectE": 2}, 210],
#           [{"objectA": 1, "objectB": 1, "objectC": 0, "objectD": 2, "objectE": 0}, 90],
#           [{"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 0, "objectE": 0}, 120]]
# # Two random number generator streams are fixed so that comparisons with other stochastic methods are more accurate
# rg_move_stream = random.Random(1609)
# rg_throw_stream = random.Random(793)

# LARGE instance:
# ORDERS I, tot obj 9 + 11 + 11 = 31
orders = [[{"objectA": 3, "objectB": 2, "objectC": 1, "objectD": 2, "objectE": 1}, 310],
          [{"objectA": 2, "objectB": 2, "objectC": 4, "objectD": 3, "objectE": 0}, 550],
          [{"objectA": 1, "objectB": 2, "objectC": 3, "objectD": 2, "objectE": 3}, 290]]
rg_move_stream = random.Random(2404)
rg_throw_stream = random.Random(610)

# Sorting the orders in ascending order of priority (highest priority = lowest delivery time)
orders.sort(key=lambda x: x[1])

initial_len_orders = len(orders)
mission = {}
trays_delivery_times = []
trays_entering_times = []
mission['tray0'] = orders[0][0]
trays_delivery_times.append(orders[0][1])
trays_entering_times.append(0)
orders.remove(orders[0])

# Defined by me
maxPortableObjs = 4
rewards = {'move': 0, 'pick': 10, 'place': 12}

# I can only place or throw from placing nodes (to any tray)
placing_nodes = nodes[-1]


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
    # Returns the new state after placing action a (= [index of object type to place: A=0, B=1, etc., index of tray]),
    # given current state s
    new_s = s.copy()
    new_s[0] += tthrow
    new_s[3 + (T+1) * a[0] + a[1]] += 1
    return new_s


# ########################################### ADDITIONAL NEEDED FUNCTIONS ########################################### #

def tot_objects_placed(s):
    # Given the state s of the system, returns a list where each entry j specifies how many objects of type j
    # have already been placed
    placed = [0] * O
    for j in range(len(placed)):
        for t in range(T):
            placed[j] += s[3 + (T+1) * j + t]
    return placed


def objects_collected(s):
    # Yields the number of objects collected but not yet placed (inferred from the state values)
    # INPUT
    # s: current state of the system
    # OUTPUT
    # obj: the number of objects collected but not yet placed in the given state
    objs = 0
    placed = tot_objects_placed(s)
    for j in range(O):
        objs += s[2 + (T+1) * j] - placed[j]  # picked - placed
    return objs


# ################################################# MAIN EXECTUTION ################################################# #

start = timeit.default_timer()
time.sleep(1)

for o in range(initial_len_orders):
    actions = []
    states = []
    if o == 0:
        # First order
        init = [0] * (O * (T + 1) + 2)
        # init[1] = random.choice(nodes)
        init[1] = 'np0'
        states.append(init)
        current_state = init.copy()
    else:
        # Other orders
        init = [0] * (O * (T + 1) + 2)
        init[0] = old_states[-1][0]  # Re-start from time of previous order completion
        init[1] = old_states[-1][1]  # Re-start from node of previous order completion
        states.append(init)
        current_state = init.copy()
    if init[1] != 'np0':
        # Always (re)start from np0
        current_state = new_state_moving('np0', init)
        p_risk = nodes_connections[(init[1], 'np0')]['risk'] / 100
        action = 'move np0'
        if rg_move_stream.random() < p_risk:
            current_state[0] += 5  # reach next state but with 5 second penalty
            action = action + ' ' + '(COLLISION)'
        states.append(current_state)
        actions.append(action)
    j = 0
    while j < O:
        # Until all objects of type j are picked or the robot has reached maximum capacity
        while current_state[2 + (T+1) * j] < mission['tray0'][objects[j]] and \
                objects_collected(current_state) < maxPortableObjs:
            next_state = new_state_picking(j, current_state)
            states.append(next_state)
            actions.append('pick' + ' ' + str(j))
            current_state = next_state.copy()
        if objects_collected(current_state) < maxPortableObjs and j < O - 1:
            # If the robot has not reached maximum capacity move to the next box (if there is one)
            next_state = new_state_moving(nodes[j + 1], current_state)
            action = 'move' + ' ' + nodes[j + 1]
            p_risk = nodes_connections[(current_state[1], nodes[j+1])]['risk'] / 100
            if rg_move_stream.random() < p_risk:
                next_state[0] += 5  # Reach next state but with 5 second penalty
                action = action + ' ' + '(COLLISION)'
            states.append(next_state)
            actions.append(action)
            current_state = next_state.copy()
        else:
            # Move to the placing location to throw the objects collected so far
            next_state = new_state_moving(placing_nodes, current_state)
            action = 'move' + ' ' + placing_nodes
            p_risk = nodes_connections[(current_state[1], placing_nodes)]['risk'] / 100
            if rg_move_stream.random() < p_risk:
                next_state[0] += 5  # Reach next state but with 5 second penalty
                action = action + ' ' + '(COLLISION)'
            states.append(next_state)
            actions.append(action)
            current_state = next_state.copy()
            for i in range(j+1):  # For all object types collected so far
                if current_state[2 + (T+1) * i] != current_state[3 + (T+1) * i]:
                    # If the robot has some objects of type i to place
                    for k in range(current_state[3 + (T+1) * i], current_state[2 + (T+1) * i]):
                        # Place all the collected objects of type i
                        next_state = new_state_throwing([i, 0], current_state)
                        action = 'throw' + ' ' + str(i)
                        p_success = throwing_success(nodes_coordinates[current_state[1]], trays_coordinates['tray0'])
                        if rg_throw_stream.random() > p_success:
                            next_state[2 + (T + 1) * k[0]] -= 1  # Throwing failed: object lost
                            next_state[3 + (T + 1) * k[0]] -= 1  # Remove object counted as placed
                            action = action + ' ' + '(FAILED)'
                        states.append(next_state)
                        actions.append(action)
                        current_state = next_state.copy()
            if tot_objects_placed(current_state) != list(mission['tray0'].values()):  # Order not completed
                if tot_objects_placed(current_state)[j] != list(mission['tray0'].values())[j]:
                    # If there are remaining objects of type j to pick
                    # Go back to the last box the robot was at before going to placing node
                    next_state = new_state_moving(nodes[j], current_state)
                    action = 'move' + ' ' + nodes[j]
                    p_risk = nodes_connections[(current_state[1], nodes[j])]['risk'] / 100
                    if rg_move_stream.random() < p_risk:
                        next_state[0] += 5  # Reach next state but with 5 second penalty
                        action = action + ' ' + '(COLLISION)'
                    states.append(next_state)
                    actions.append(action)
                    current_state = next_state.copy()
                    j -=1
                else:
                    # Go to the next box wrt the one the robot was at before going to placing node
                    next_state = new_state_moving(nodes[j+1], current_state)
                    action = 'move' + ' ' + nodes[j+1]
                    p_risk = nodes_connections[(current_state[1], nodes[j+1])]['risk'] / 100
                    if rg_move_stream.random() < p_risk:
                        next_state[0] += 5  # Reach next state but with 5 second penalty
                        action = action + ' ' + '(COLLISION)'
                    states.append(next_state)
                    actions.append(action)
                    current_state = next_state.copy()
        if tot_objects_placed(current_state) == list(mission['tray0'].values()):
            break
        j += 1

    print(actions)
    print(states)

    old_states = states.copy()

    if len(orders) != 0:
        # Change order and start again
        mission['tray0'] = orders[0][0]
        orders.remove(orders[0])

end = timeit.default_timer()

print(f"Time taken is {end - start}s")
