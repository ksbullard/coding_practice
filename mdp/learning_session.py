import numpy as np
from representation.environment import Grid, GridWorld


def main():
    print "Let's get started with MDP's...."
    start_position, W = (0, 0), Grid.wall_value()
    grid = np.array(
                [[0,0,0,0],
                 [0, W, 0, -1],
                 [0,0,0,1]]
            )
    environment = GridWorld(grid, start_position)





if __name__ == "__main__":
    main()