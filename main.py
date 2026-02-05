import numpy as np

from maze.grid import Grid, SMALL, MEDIUM, LARGE

g1 = Grid(SMALL)
g2 = Grid(MEDIUM)
g3 = Grid(LARGE)


from maze.grid import Grid, SMALL

g = Grid(SMALL, seed=123)
print(g.base_cost[g.start], g.base_cost[g.goal])