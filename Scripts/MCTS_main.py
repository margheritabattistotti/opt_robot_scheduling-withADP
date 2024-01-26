import math
import random
import timeit
import time
import Myopic_Rollout as MR
import MCTree as MT


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


# ################################################# MAIN EXECUTION ################################################# #

exec_time = []
objective_values = []

for iters in range(15):

    start = timeit.default_timer()
    time.sleep(1)

    initial_state = [0] * (MR.O * (MR.T + 1) + 2)
    initial_state[1] = 'np0'


    def forward_pass(initial_s):
        # Defines the optimal strategy to complete the mission following a MonteCarlo Tree Search starting
        # from 'initial_s',
        # OUTPUT
        # action_sequence: sequence of optimal actions
        # states_sequence: optimal state sequence
        s = MT.MCTSPreDecisionNode(initial_s)
        objective = 0
        action_sequence = []
        state_sequence = [s.state]
        while not MR.is_terminal(s.state):
            best_action_type, best_a = s.MCTS_best_action()
            if best_action_type == 'throw':
                best_action = best_action_type + ' ' + str(best_a)
                p_success = throwing_success(MR.nodes_coordinates[s.state[1]], MR.trays_coordinates[MR.trays[best_a[1]]])
                if random.random() > p_success:
                    best_new_state = MR.failed_new_state_throwing(best_a, s.state)
                    best_action = best_action + ' ' + '(FAILURE)'
                else:
                    best_new_state = MR.new_state_throwing(best_a, s.state)
            elif best_action_type == 'move':
                best_action = best_action_type + ' ' + str(best_a)
                p_risk = MR.nodes_connections[(s.state[1], best_a)]['risk'] / 100
                if random.random() < p_risk:
                    best_new_state = MR.failed_new_state_moving(best_a, s.state)
                    best_action = best_action + ' ' + '(COLLISION)'
                else:
                    best_new_state = MR.new_state_moving(best_a, s.state)
            else:
                best_action = best_action_type + ' ' + str(best_a)
                best_new_state = MR.new_state_picking(best_a, s.state)

            state_sequence.append(best_new_state)
            action_sequence.append(best_action)
            s = MT.MCTSPreDecisionNode(best_new_state)

        objective += MR.terminal_state_evaluation(s.state)

        return action_sequence, state_sequence, objective


    actions_seq, states_seq, objective_value = forward_pass(initial_state)

    objective_values.append(objective_value)

    print(actions_seq)
    print(states_seq)
    print(objective_value)

    end = timeit.default_timer()

    exec_time.append(end - start)

    # print(f"Time taken is {end - start}s")

print(sum(objective_values)/len(objective_values))
print(f"Mean execution time is {sum(exec_time)/15}")
