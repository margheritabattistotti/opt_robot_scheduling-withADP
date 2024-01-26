import math
import random


# ########################################  GLOBAL VARIABLES DEFINITION  ######################################## #

tpick = 7  # (s) time elapsed to perform picking action
tthrow = 5  # (s) time elapsed to perform throwing action
maxPortableObjs = 4
# rewards = {'move': 0, 'pick': 0.8, 'throw': 1}
# rewards = {'move': 0, 'pick': 10, 'throw': 12}
rewards = {'move': 0, 'pick': 20, 'throw': 25}


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


# #######################################  SIMULATOR CLASS DEFINITION  ######################################## #

class Simulator:

    def __init__(self, initial_state, nodes, nodes_coordinates, throwing_nodes, objects, objects_pick_nodes, trays,
                 trays_coordinates, nodes_connections, mission, Thorizon):
        self.initial_state = initial_state
        self.current_state = self.initial_state.copy()
        self.post_decision_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.nodes = nodes
        self.nodes_coordinates = nodes_coordinates
        self.throwing_nodes = throwing_nodes
        self.objects = objects
        self.O = len(self.objects)
        self.objects_pick_nodes = objects_pick_nodes
        self.trays = trays
        self.T = len(self.trays)
        self.trays_coordinates = trays_coordinates
        self.nodes_connections = nodes_connections
        self.mission = mission
        self.Thorizon = Thorizon
        self.tot_obj4mission = self.tot_objects4mission()
        self.terminal = False

    # ###################################  SIMULATION OF NEXT STATE  ################################### #

    def simulation(self, action_type, a):
        failure = False
        self.current_state = self.next_state.copy()
        reward = rewards[action_type]
        if action_type == 'move':
            p_risk = self.nodes_connections[(self.current_state[1], self.post_decision_state[1])]['risk'] / 100
            self.next_state = self.post_decision_state.copy()
            if random.random() < p_risk:
                failure = True
                self.next_state[0] += 5  # Reach next state but with 5 second penalty
                reward = -2
        elif action_type == 'throw':
            reward *= (self.Thorizon * 2 - self.current_state[0]) / self.Thorizon
            p_success = throwing_success(self.nodes_coordinates[self.current_state[1]],
                                         self.trays_coordinates[self.trays[a[1]]])
            self.next_state = self.post_decision_state.copy()
            if random.random() > p_success:
                failure = True
                # Object just placed in post-decision state actually isn't placed
                self.next_state[3 + (self.T + 1) * a[0] + a[1]] -= 1
                # Object just thrown actually lost from picked objects
                self.next_state[2 + (self.T + 1) * a[0]] -= 1
                reward = 0
        elif action_type == 'pick':
            reward *= (self.Thorizon * 2 - self.current_state[0]) / self.Thorizon
            self.next_state = self.post_decision_state.copy()

        self.is_terminal()

        return self.current_state, action_type, a, self.post_decision_state, failure, \
            self.next_state, reward, self.terminal

    # ########################################  TERMINAL STATE CHECK  ######################################## #

    def is_terminal(self):
        if self.next_state[0] >= self.Thorizon or \
                self.tot_objects_placed() == self.tot_obj4mission:
            self.terminal = True

    # ##################################### TERMINAL VALUE FUNCTION DEFINITION ##################################### #

    # # Terminal value function I
    # def terminal_value(self):
    #     # Value function for a terminal state s
    #     V = float(self.Thorizon - self.next_state[0])
    #     placed = self.tot_objects_placed()
    #     for j in range(self.O):
    #         V -= self.tot_obj4mission[j] - placed[j]
    #         V += float(self.next_state[2 + (self.T + 1) * j])
    #     return V

    # # Terminal value function II
    # def terminal_value(self):
    #     # Value function for a terminal state s
    #     V = 0.5 * (self.Thorizon - self.next_state[0])
    #     placed = self.tot_objects_placed()
    #     for j in range(self.O):
    #         V -= 15 * (self.tot_obj4mission[j] - placed[j])
    #     return V

    # Terminal value function III
    def terminal_value(self):
        # Value function for a terminal state s
        V = 5 * (self.Thorizon - self.next_state[0])
        placed = self.tot_objects_placed()
        for j in range(self.O):
            V -= rewards['throw'] * (self.tot_obj4mission[j] - placed[j])
            V += rewards['pick'] * self.next_state[2 + (self.T + 1) * j]
        return V

    # ###################################  RESET STATES FOR A NEW ITERATION  ################################### #

    def reset(self):
        self.current_state = self.initial_state.copy()
        self.post_decision_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.terminal = False

    # ###################################### OTHER SPECIFIC METHODS NEEDED ###################################### #

    def tot_objects4mission(self):
        # Returns a list where each entry j specifies how many objects of type j must be placed during mission
        tot_object = [0] * self.O
        for j in range(self.O):
            for t in range(self.T):
                tot_object[j] += list(self.mission.values())[t][self.objects[j]]

        return tot_object

    def tot_objects_placed(self):
        # Given the state of the system, returns a list where each entry j specifies how many objects of type j
        # have already been placed
        placed = [0] * self.O
        for j in range(self.O):
            for t in range(self.T):
                placed[j] += self.next_state[3 + (self.T + 1) * j + t]

        return placed
