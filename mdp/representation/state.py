import numpy as np
import environment as env


class State:
    def __init__(self, x=0, y=0, is_start=False, is_terminal=False, is_wall=False):
        self.__position = env.Position(x,y)
        self.__is_start = is_start
        self.__has_wall = is_wall
        self.__is_terminal = is_terminal


    def __eq__(self, other):
        return isinstance(other, State) and other.has_same_position(self)


    def __str__(self):
        return str(self.__position)



    ##### GETTERS #####
    def position(self):
        return self.__position

    def is_start(self):
        return self.__is_start

    def is_terminal(self):
        return self.__is_terminal

    def has_wall(self):
        return self.__has_wall


    ##### SETTERS #####
    def set_is_start_state(self, flag):
        self.__is_start = flag

    def set_is_terminal_state(self, flag):
        self.__is_terminal = flag

    def set_has_wall(self, flag):
        self.__has_wall = flag


    ##### OTHER METHODS #####
    def has_same_position(self, other):
        if isinstance(other, State):
            return (self.position().x() == other.position().x()) and \
                   (self.position().y() == other.position().y())
        elif isinstance(other, env.Position):
            return (self.position().x() == other.x()) and \
                   (self.position().y() == other.y())
        elif (isinstance(other, list) or isinstance(other, tuple) or isinstance(other, np.ndarray)) \
                and len(other) == 2:
            return (self.position().x() == other[0]) and \
                   (self.position().y() == other[1])
        elif isinstance(other, dict) and ('x' in other) and ('y' in other):
            return (self.position().x() == other['x']) and \
                   (self.position().y() == other['y'])
        return False