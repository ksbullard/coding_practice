import numpy as np
from mdp import MarkovDecisionProcess
from environment import Position
from action import ActionType, Action



class GridworldMDP(MarkovDecisionProcess):
    def __init__(self, states=None, actions=None):
        self.__states = states
        if self.__states is None:
            self.__states = []
        self.__actions = actions
        if self.__actions is None:
            self.__actions = [Action(ActionType.UP), Action(ActionType.DOWN), Action(ActionType.LEFT), Action(ActionType.RIGHT)]

        self.__action_cost = -0.04



    def gamma(self):
        """Return discount factor"""
        return 1.0  # 0.5


    def get_states(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        return self.__states

    def get_start_state(self):
        """
        Return the start state of the MDP.
        """
        for s in self.__states:
            if s.is_start():
                return s
        return None

    def get_possible_actions(self, state):
        """
        Return list of possible actions from 'state'.
        """
        if state.is_terminal() or state.has_wall():
            return []
        else:
            return self.__actions


    def get_transition_states_and_probs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        Note that in Q-Learning and reinforcment
        learning in general, we do not know these
        probabilities nor do we directly model them.
        """

        transitions = []

        # first -- check absorbing states
        if state.is_terminal():
            return transitions

        # intended action
        next_state = self.get_next_state(state, action)
        prob = 0.8
        transitions.append((next_state, prob))

        # unintended actions
        prob = 0.1
        if action.type() == ActionType.UP or action.type() == ActionType.DOWN:
            # left
            next_state = self.get_next_state(state, ActionType.LEFT)
            transitions.append((next_state, prob))

            # right
            next_state = self.get_next_state(state, ActionType.RIGHT)
            transitions.append((next_state, prob))

        elif action.type() == ActionType.LEFT or action.type() == ActionType.RIGHT:
            # up
            next_state = self.get_next_state(state, ActionType.UP)
            transitions.append((next_state, prob))

            # down
            next_state = self.get_next_state(state, ActionType.DOWN)
            transitions.append((next_state, prob))
        return transitions



    def get_next_state(self, state, action):
        # compute next state position
        next_state_coordinates = None
        if isinstance(action, Action):
            next_state_coordinates = np.array(state.position().xy()) + np.array(action.velocity())
        elif isinstance(action, ActionType):
            next_state_coordinates = np.array(state.position().xy()) + np.array(Action.action_velocity(action))


        if next_state_coordinates is not None:
            next_state_position = Position(next_state_coordinates[0], next_state_coordinates[1])

            '''
            # check (and correct) boundary cases
            if next_state_position.x() < 0:
                next_state_position.set_x(0)
            elif next_state_position.x() > 3:
                next_state_position.set_x(3)

            if next_state_position.y() < 0:
                next_state_position.set_y(0)
            elif next_state_position.y() > 2:
                next_state_position.set_y(2)
            '''

            # check (and correct) boundary cases or wall state(s)
            if next_state_position.x() < 0 or next_state_position.x() > 3 or next_state_position.y() < 0 or \
                    next_state_position.y() > 2 or (next_state_position.x() == 1 and next_state_position.y() == 1):
                return state


            # find state with corresponding position
            for s in self.__states:
                if s.has_same_position(next_state_position):
                    return s
        return None



    def get_reward(self, state): #, action, nextState
        """
        Get the reward for the state, action, nextState transition.
        Not available in reinforcement learning.
        """
        if state.is_terminal():
            if state.has_same_position(Position(3,2)):
                return +1
            elif state.has_same_position(Position(3,1)):
                return -1
        return self.__action_cost  # else



    def is_terminal(self, state):
        """
        Returns true if the current state is a terminal state.  By convention,
        a terminal state has zero future rewards.  Sometimes the terminal state(s)
        may have no possible actions.  It is also common to think of the terminal
        state as having a self-loop action 'pass' with zero reward; the formulations
        are equivalent.
        """
        return state.is_terminal()