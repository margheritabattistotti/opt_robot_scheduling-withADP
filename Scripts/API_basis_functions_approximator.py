import numpy as np


class BasisFs:

    def __init__(self, objects, trays, nodes, throwing_nodes, mission, Thorizon, parameters, discount_factor):
        self.objects = objects
        self.O = len(objects)
        self.trays = trays
        self.T = len(trays)
        self.throwing_nodes = throwing_nodes
        self.nodes = nodes
        self.mission = mission
        self.Thorizon = Thorizon
        self.tot_obj4mission = self.tot_objects4mission()
        self.old_thetas = parameters
        self.new_thetas = parameters
        self.lmbd = 1   # or = discount_factor
        self.gamma = None
        self.B = 0.05 * np.eye(len(parameters))
        self.H = None
        self.err = None

    def get_value(self, s):
        return np.dot(self.old_thetas, self.vector_phi(s))

    # ######################################## BASIS FUNCTIONS DEFINITION ######################################## #

    # def phi_1(self, s):
    #     # Time
    #     value = self.Thorizon - s[0]
    #     return value
    #
    # def phi_2(self, s):
    #     # Total number of picked objects
    #     return sum(s[2 + (self.T+1) * o] for o in range(self.O))
    #
    # def phi_3(self, s):
    #     # Total number of not placed objects
    #     placed = self.tot_objects_placed(s)
    #     return sum(self.tot_obj4mission[o] - placed[o] for o in range(self.O))

    # def phi_4(self, s):
    #     # Is the robot in a throwing node? (binary: ocho)
    #     if s[1] in self.throwing_nodes and self.objects_collected(s) == 4:
    #         value = 1
    #     else:
    #         value = 0
    #     return value

    # def trays_level(self, s):
    #     # Given the state s of the system, returns a list where each entry t specifies how many objects
    #     # have already been placed in tray t
    #     placed = [0] * self.T
    #     for t in range(self.T):
    #         for j in range(self.O):
    #             placed[t] += s[3 + (self.T + 1) * j + t]
    #     return placed

    def phi_0(self, s):
        # Defines the value of the intercept of the linear approximation through an indicator function depending on the
        # current time elapsed from the beginning of the mission
        value = [0, 0, 0]  # 3 different intercepts for 3 different dicretized periods of time
        if 0 <= s[0] < self.Thorizon/3:
            value[0] = 1
        elif self.Thorizon/3 <= s[0] < 2 * self.Thorizon/3:
            value[1] = 1
        else:
            value[2] = 1

        return value

    def phi_4(self, s):
        # Defines the value of an independent variable accounting for the importance of the robot being in a throwing
        # node, for every throwing node, through an indicator function
        value = [0] * len(self.throwing_nodes)
        for t in range(len(self.throwing_nodes)):
            if s[1] in self.throwing_nodes and (self.objects_collected(s) == 4 or
                                                (self.tot_objects_picked(s) == self.tot_obj4mission and
                                                self.tot_objects_placed(s)[t] < self.tot_obj4mission[t])):
                # If the robot is in a placing location and (the maximum capacity is reached or
                # (the robot has already picked all objects needed for mission but still needs to place some of them))
                value[t] = 1
            else:
                value[t] = 0
        return value

    def phi_5(self, s):
        # Defines the value of an independent variable accounting for the importance of the robot being in a picking
        # node, for every picking node, through an indicator function
        value = [0] * self.O
        for o in range(self.O):
            if s[1] == self.nodes[o] and self.objects_collected(s) < 4 \
                    and s[2 + (self.T + 1) * o] < self.tot_obj4mission[o]:
                # If the robot is in a picking location and has not reached maximum capacity and some objects of type o
                # still need to be picked from the respective box
                value[o] = 1
        return value

    def nodes_one_hot(self, node):
        # Returns a unique numerical encoding for a node, because a string cannot be used for regression
        encoding = [0] * len(self.nodes)
        encoding[self.nodes.index(node)] = 1
        return encoding

    def vector_phi(self, s):
        # Defines the vector of all independent variables as defined above by the basis functions
        x = s.copy()
        node = x[1]
        x.remove(x[1])
        x.extend(self.nodes_one_hot(node))
        x.extend(self.phi_4(s))
        x.extend(self.phi_5(s))
        return self.phi_0(s) + x

    # ############################################ PARAMETERS UPDATE ############################################ #

    def model_update(self, s, v):
        # Updates the model parameters through the Recursive Least Squares method given a state s and its actual value
        phi_vect = self.vector_phi(s)
        self.gamma = self.lmbd + np.dot(np.dot(phi_vect, self.B), np.transpose(phi_vect))
        self.H = 1 / self.gamma * self.B
        self.B = 1/self.lmbd * (self.B - 1/self.gamma * np.dot(np.dot(self.B,
                                np.outer(phi_vect, phi_vect)), self.B))
        self.err = np.dot(self.new_thetas, self.vector_phi(s)) - v

        self.new_thetas = self.new_thetas - np.dot(self.H, phi_vect) * self.err

    # ###################################### OTHER SPECIFIC METHODS NEEDED ###################################### #

    def tot_objects4mission(self):
        # Returns a list where each entry j specifies how many objects of type j must be placed during mission
        tot_object = [0] * self.O
        for j in range(self.O):
            for t in range(self.T):
                tot_object[j] += list(self.mission.values())[t][self.objects[j]]

        return tot_object

    def objects_collected(self, s):
        # Yields the number of objects collected but not yet placed (inferred from the state entries)
        objs = 0
        placed = self.tot_objects_placed(s)
        for j in range(self.O):
            objs += s[2 + (self.T + 1) * j] - placed[j]  # picked - placed
        return objs

    def tot_objects_placed(self, s):
        # Given the state of the system, returns a list where each entry j specifies how many objects of type j
        # have already been placed
        placed = [0] * self.O
        for j in range(self.O):
            for t in range(self.T):
                placed[j] += s[3 + (self.T + 1) * j + t]
        return placed

    def tot_objects_picked(self, s):
        # Given the state of the system, returns a list where each entry j specifies how many objects of type j
        # have already been placed
        picked = [0] * self.O
        for j in range(self.O):
            for t in range(self.T):
                picked[j] += s[2 + (self.T + 1) * j]
        return picked
