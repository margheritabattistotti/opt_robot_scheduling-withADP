import math
import random
from API_basis_functions_approximator import BasisFs


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


# #######################################  APPROXIMATOR CLASS DEFINITION  ######################################## #

class Approximator:

    def __init__(self, initial_state, nodes, nodes_coordinates, throwing_nodes, objects, objects_pick_nodes, trays,
                 trays_coordinates, nodes_connections, mission, Thorizon, greedy_epsilon, discount_factor, parameters):
        self.initial_state = initial_state
        self.current_state = self.initial_state.copy()
        self.post_decision_state = self.initial_state.copy()
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
        self.epsilon = greedy_epsilon
        self.gamma = discount_factor
        self.model = BasisFs(objects, trays, nodes, throwing_nodes, mission, Thorizon, parameters, discount_factor)

    # ######################################  ADMISSIBLE ACTIONS DEFINITION ###################################### #

    # Moving is allowed from each node to all others. Info on time and risk in nodes_connections.
    def moving_actions(self):
        # Returns all possible moving actions from the current state
        # OUTPUT
        # mov_actions: list containing all possible moving actions, identified by the destination node
        move_actions = []
        all_destinations = self.nodes.copy()
        all_destinations.remove(self.current_state[1])
        for k in all_destinations:
            if self.current_state[0] + self.nodes_connections[(self.current_state[1], k)]['time'] <= self.Thorizon:
                move_actions.append(k)

        if len(move_actions) == 0:
            # No moving actions allowed
            move_actions = None

        return move_actions

    # Picking is only allowed in picking nodes, and the robot can only pick the object in the respective box
    def picking_actions(self):
        # Returns the possible picking action from the current state
        # OUTPUT
        # pick_action: picking action identified by the type of object to pick (A=0, B=1, C=2, etc.)
        pick_action = []
        if self.current_state[0] + tpick <= self.Thorizon:
            for j in range(self.O):
                if list(self.objects_pick_nodes.values())[j] == self.current_state[1] \
                        and self.current_state[2 + (self.T + 1) * j] < self.tot_obj4mission[j] \
                        and self.objects_collected() < maxPortableObjs:
                    pick_action.append(j)

        return pick_action

    # Throwing is allowed only in placing nodes
    def throwing_actions(self):
        # Returns all possible throwing actions from the current state
        # OUTPUT
        # throw_actions: list containing all possible throwing actions, identified by the object type to place and the
        # tray where to throw it (A=0, B=1, etc. and trays as tray0=0, tray1=1. etc.)
        throw_actions = []
        if self.current_state[1] not in self.throwing_nodes:
            # Robot not in a throwing location
            return None
        if self.current_state[0] + tthrow > self.Thorizon:
            # No time to perform action
            return None
        for t in range(self.T):
            for j in range(self.O):
                placed = self.tot_objects_placed()[j]
                # picked > placed & mission not yet completed
                if self.current_state[2 + (self.T + 1) * j] > placed \
                        and self.current_state[3 + (self.T+1) * j + t] < list(self.mission[self.trays[t]].values())[j]:
                    throw_actions.append([j, t])

        if len(throw_actions) == 0:
            # No throwing actions allowed
            throw_actions = None

        return throw_actions

    def all_admissible_actions(self):
        # Returns all admissible actions from the current state
        actions = {}

        move = self.moving_actions()
        pick = self.picking_actions()
        throw = self.throwing_actions()

        if move is not None:
            actions['move'] = move

        if len(pick) != 0:
            actions['pick'] = pick

        if throw is not None:
            actions['throw'] = throw

        return actions

    # ############################## TRANSITIONS TO POST DECISION STATES DEFINITION ############################## #

    def new_state_moving(self, a):
        # Returns the new state after moving action from the current state to the given destination a, exploiting
        # information on their connection {'time': , 'risk': , 'path': }
        s = self.current_state.copy()
        s[0] += self.nodes_connections[(self.current_state[1], a)]['time']
        s[1] = a

        return s

    def failed_new_state_moving(self, a):
        # Returns the new state when moving action from the current state to the given destination a fails
        s = self.current_state.copy()
        s[0] += self.nodes_connections[(self.current_state[1], a)]['time']
        if s[0] + 5 <= self.Thorizon:
           s[0] += 5
        else:
            s[0] = self.Thorizon
        s[1] = a

        return s

    def new_state_picking(self, a):
        # Returns the new state after picking action a (= index of object type to pick : A=0, B=1, etc.),
        # from the current state
        s = self.current_state.copy()
        s[0] += tpick
        s[2 + (self.T + 1) * a] += 1

        return s

    def new_state_throwing(self, a):
        # Returns new state after throwing action a (= [index of object type to place: A=0, B=1, etc., index of tray])
        # from current state
        s = self.current_state.copy()
        s[0] += tthrow
        s[3 + (self.T + 1) * a[0] + a[1]] += 1

        return s

    def failed_new_state_throwing(self, a):
        # Returns new state when throwing action a (= [index of object type to place: A=0, B=1, etc., index of tray])
        # fails from current state
        s = self.current_state.copy()
        s[0] += tthrow
        s[2 + (self.T + 1) * a[0]] -= 1

        return s

    # ########################################### CHOOSING BEST ACTION ########################################### #

    def get_action(self):
        max_value = -100000
        best_action_type = None
        best_action = None
        actions = self.all_admissible_actions()
        if len(actions) == 0:  # No admissible actions from current state
            return None, None, None, None, None
        if self.epsilon > random.random():  # Exploration
            best_action_type = random.choice(list(actions.keys()))
            best_action = random.choice(actions[best_action_type])
            if best_action_type == 'move':
                s = self.new_state_moving(best_action)
                failed_s = self.failed_new_state_moving(best_action)
                p_risk = self.nodes_connections[(self.current_state[1], s[1])]['risk'] / 100
                # Expected value
                max_value = self.gamma * ((1-p_risk) * (self.model.get_value(s) + rewards['move'])
                                          + p_risk * (self.model.get_value(failed_s) - 2))
                self.post_decision_state = s.copy()
            elif best_action_type == 'pick':
                s = self.new_state_picking(best_action)
                max_value = self.gamma * self.model.get_value(s) + rewards['pick'] \
                            * (self.Thorizon * 2 - s[0]) / self.Thorizon
                self.post_decision_state = s.copy()
            elif best_action_type == 'throw':
                s = self.new_state_throwing(best_action)
                failed_s = self.failed_new_state_throwing(best_action)
                p_success = throwing_success(self.nodes_coordinates[self.current_state[1]],
                                             self.trays_coordinates[self.trays[best_action[1]]])
                # Expected value
                max_value = self.gamma * (p_success * (self.model.get_value(s) + rewards['throw'] *
                                                       (self.Thorizon * 2 - s[0]) / self.Thorizon) +
                                                      (1-p_success) * self.model.get_value(failed_s))
                self.post_decision_state = s.copy()
        else:  # Exploitation
            for key in actions.keys():
                for a in actions[key]:
                    if key == 'move':
                        s = self.new_state_moving(a)
                        failed_s = self.failed_new_state_moving(a)
                        p_risk = self.nodes_connections[(self.current_state[1], s[1])]['risk'] / 100
                        # Expected value
                        temp_value = self.gamma * ((1 - p_risk) * (self.model.get_value(s) + rewards['move'])
                                                   + p_risk * (self.model.get_value(failed_s) - 2))
                        if temp_value > max_value:
                            max_value = temp_value
                            best_action_type = key
                            best_action = a
                            self.post_decision_state = s.copy()
                    elif key == 'pick':
                        s = self.new_state_picking(a)
                        temp_value = self.gamma * self.model.get_value(s) + rewards['pick'] * \
                                     (self.Thorizon * 2 - self.current_state[0]) / self.Thorizon
                        if temp_value > max_value:
                            max_value = temp_value
                            best_action_type = key
                            best_action = a
                            self.post_decision_state = s.copy()
                    elif key == 'throw':
                        s = self.new_state_throwing(a)
                        failed_s = self.failed_new_state_throwing(a)
                        p_success = throwing_success(self.nodes_coordinates[self.current_state[1]],
                                                     self.trays_coordinates[self.trays[a[1]]])
                        # Expected value
                        temp_value = self.gamma * (
                                    p_success * (self.model.get_value(s) + rewards['throw'] * (self.Thorizon * 2 - s[0])
                                                 / self.Thorizon) + (1-p_success) * self.model.get_value(failed_s))
                        if temp_value > max_value:
                            max_value = temp_value
                            best_action_type = key
                            best_action = a
                            self.post_decision_state = s.copy()

        return self.current_state, best_action_type, best_action, max_value, self.post_decision_state

    # ###################################  RESET STATES FOR A NEW ITERATION  ################################### #

    def reset(self):
        self.current_state = self.initial_state.copy()
        self.post_decision_state = self.initial_state.copy()

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
                placed[j] += self.current_state[3 + (self.T + 1) * j + t]
        return placed

    def objects_collected(self):
        # Yields the number of objects collected but not yet placed (inferred from the state entries)
        objs = 0
        placed = self.tot_objects_placed()
        for j in range(self.O):
            objs += self.current_state[2 + (self.T + 1) * j] - placed[j]  # picked - placed
        return objs
