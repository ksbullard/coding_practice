import numpy as np
from mdp.representation.gridworld_mdp import GridworldMDP
from mdp.representation.state import State



def value_iteration(mdp):

    epsilon = 1e-4
    convergence_threshold = (epsilon * (1 - mdp.gamma())) / mdp.gamma()
    t = 0
    should_terminate = False
    prev_val, val, pi = {}, {}, {}

    while not should_terminate:  #for t in range(num_time_steps):
        for s in mdp.get_states():
            if t == 0:
                prev_val[s.position()] = 0
                if s.is_terminal():
                    prev_val[s.position()] = mdp.get_reward(s)
            else:
                action_values = {} #np.array(len(mdp.get_possible_actions()))
                for a in mdp.get_possible_actions(s):
                    action_transitions = mdp.get_transition_states_and_probs(s, a)
                    action_values[a] = 0
                    for transition in action_transitions:
                        next_state, trans_prob = transition[0], transition[1]
                        action_values[a] += trans_prob * prev_val[next_state.position()]

                # take value associated with arg max action
                val[s.position()] = mdp.get_reward(s)
                pi[s.position()] = []
                if len(action_values) > 0:
                    max_action_value = max(action_values.values())
                    val[s.position()] += mdp.gamma() * max_action_value
                    pi[s.position()] = [k for k, v in action_values.items() if v == max_action_value]
                pass


        if t > 0:
            # update previous values dictionary with the most current computed values
            min_time = 2
            if t > min_time:
                should_terminate = True
            for s in mdp.get_states():
                if t > min_time:
                    if np.abs(val[s.position()] - prev_val[s.position()]) > convergence_threshold:
                        should_terminate = False

                prev_val[s.position()] = val[s.position()]
        t += 1
    return val, pi


def monte_carlo(mdp):

    # TODO: Implement MC online learning method for solving mdp
    pass



def temporal_differencing(mdp):

    # TODO: Implement sarsa and q learning versions of solving grid world mdp
    pass




def main():
    world_states = []
    # Create grid world
    for i in range(4):
        for j in range(3):
            if i == 3 and (j == 1 or j == 2):
                state = State(i, j, is_terminal=True)
            elif i == 0 and j == 0:
                state = State(i, j, is_start=True)
            elif i == 1 and j == 1:
                state = State(i, j, is_wall=True)
            else:
                state = State(i, j)
            world_states.append(state)


    gridworld_task = GridworldMDP(states=world_states)
    state_values, optimal_policy = value_iteration(gridworld_task)
    pass



if __name__ == '__main__':
    main()