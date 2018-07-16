from enum import Enum


class ActionType(Enum):
    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, Enum):
            return self.value == other.value
        else:
            return self.value == other

    LEFT = 'left'
    RIGHT = 'right'
    UP = 'up'
    DOWN = 'down'


class Action:
    def __init__(self, type):
        self.__type = type

    def __str__(self):
        return str(self.__type)



    ##### GETTERS #####
    def label(self):
       str(self.__type)

    def type(self):
        return self.__type



    ##### OTHER METHODS #####
    def velocity(self):
       return Action.action_velocity(self.type())


    @staticmethod
    def action_velocity(type):
        if type == ActionType.LEFT:
            return [-1, 0]
        elif type == ActionType.RIGHT:
            return [1, 0]
        elif type == ActionType.UP:
            return [0, 1]
        elif type == ActionType.DOWN:
            return [0, -1]
