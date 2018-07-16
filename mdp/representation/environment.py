import numpy as np
import mdp
from state import State
from action import Action



class Position:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y


    def __str__(self):
        return '(' + str(self.__x) + ',' + str(self.__y) + ')'


    def x(self):
        return self.__x


    def y(self):
        return self.__y


    def xy(self):
        """Returns position tuple <x,y>"""
        return (self.x(), self.y())




    def set_x(self, val):
        self.__x = val

    def set_y(self, val):
        self.__y = val




class Grid:
    def __init__(self, grid):
        if isinstance(grid, Grid):
            self.__grid = grid.array()
        else:
            self.__repr = grid



    ##### GETTERS #####
    def array(self):
        return self.__grid


    def width(self):
        return self.array().shape[1]


    def height(self):
        return self.array().shape[0]


    def value(self, position):
        if isinstance(position, State):
            position = position.position()

        # Get xy position
        if isinstance(position, Position):
            return self.array(position.x(), position.y())
        elif (isinstance(position, list) or isinstance(position, tuple)) and len(position) == 2:
            return self.array(position[0], position[1])
        elif isinstance(position, dict) and ('x' in position) and ('y' in position):
            return self.array(position['x'], position['y'])
        return float('nan')


    @staticmethod
    def wall_value():
        return float('nan')




class GridWorld(mdp.MarkovDecisionProcess):
    def __init__(self, grid, start_pos=None):
        self.__grid = Grid(grid)
        self.__states = []
        self.__terminal_states = []

        self.__possible_actions = [Action('up'), Action('down'), Action('left'), Action('right')]
        self.__reward_mapping = {} # state --> integer
        self.__action_ordering = ['left', 'up', 'right', 'down']

        # setting start state
        if (start_pos is not None):
            if isinstance(start_pos, Position):
                self.__start_state = State(start_pos.x(), start_pos.y(), is_start=True)
            elif (isinstance(start_pos, list) or isinstance(start_pos, tuple) or isinstance(start_pos, np.ndarray)):
                self.__start_state = State(start_pos[0], start_pos[1], is_start=True)
            elif isinstance(start_pos, dict) and ('x' in start_pos) and ('y' in start_pos):
                self.__start_state = State(start_pos['x'], start_pos['y'], is_start=True)
        else:
            self.__start_state = State(is_start=True)
        self.__reward_mapping[self.__start_state] = self.grid().value(start_pos)
        self.__curr_state = self.__start_state

        # initializing all states and placing in reward mapping
        self.set_states()



    ##### GETTERS #####
    def grid(self):
        return self.__grid


    def get_start_state(self):
        return self.__start_state


    def get_next_state(self, state, action):
        next_pos_xy = list(state.position().xy()) + action.velocity()
        if self.is_wall_position(next_pos_xy):
            next_state = state
        else:
            next_state = State(next_pos_xy[0], next_pos_xy[1])
        return next_state


    def get_possible_actions(self, state):
        return self.__possible_actions


    def get_transition_states_and_probs(self, state, action):
        next_states_with_probs = []
        k = self.action_index_from_ordering(action)
        for i in [-1, 0, 1]:
            index = (k+i) % len(self.__action_ordering)
            next_state = self.get_next_state(state, Action(self.__action_ordering[index]))
            if i == 0:
                #next_state = list(state.position().xy()) + action.velocity()
                next_states_with_probs.append((next_state, 0.8))
            else:
                next_states_with_probs.append((next_state, 0.1))

        state_counts = {}
        for s in next_states_with_probs:
            if s[0] not in state_counts:
                state_counts[s[0]] = 0
            state_counts[s[0]] += 1

        for s in state_counts:
            if state_counts[s] > 1:
                total_prob = self.find_total_prob_of_state(next_states_with_probs, s)
                self.remove_tuples_with_state(next_states_with_probs, s)
                next_states_with_probs.append(s, total_prob)
                state_counts[s] = 1
        return next_states_with_probs


    def get_reward(self, state):
        return self.grid().value(state.position())



    ##### SETTERS #####
    def set_states(self):
        for row in range(self.grid().shape[0]):
            for col in range(self.grid().shape[1]):
                is_terminal_state = self.grid().value([row, col]) is not Grid.wall_value() and self.grid().value([row, col]) != 0
                if is_terminal_state:
                    self.set_terminal_state(Position(row, col), self.grid().value([row, col]))
                elif State(row, col) != self.__start_state:
                    self.set_state(Position(row, col))



    def set_state(self, position, reward):
        term_state = State(position)
        self.__states.append(term_state)
        self.__reward_mapping[term_state] = reward



    def set_terminal_state(self, position, reward):
        term_state = State(position, is_terminal=True)
        self.__terminal_states.append(term_state)
        self.__states.append(term_state)
        self.__reward_mapping[term_state] = reward



    ##### OTHER METHODS #####
    def is_terminal(self, state):
        return state.is_terminal()


    def grid_position_exists(self, position_xy):
        x, y = position_xy[0], position_xy[1]
        return x >= 0 and x < self.grid().width() and y >= 0 and y < self.grid().height()


    def is_wall_position(self, position_xy):
        return not self.grid_position_exists(position_xy) or \
               self.__grid.array()[position_xy[0], position_xy[1]] == Grid.wall_value()


    def find_total_prob_of_state(self, state_prob_list, target_state):
        prob = 0
        for state_prob in state_prob_list:
            if state_prob[0] == target_state:
                prob += state_prob[1]
        return prob


    def remove_tuples_with_state(self, state_prob_list, target_state):
        for state_prob in state_prob_list:
            if state_prob[0] == target_state:
                state_prob_list.remove(state_prob)


    def action_index_from_ordering(self, action):
        index = 0
        for action_label in self.__action_ordering:
            if action.label() == action_label:
                return index
            index += 1



