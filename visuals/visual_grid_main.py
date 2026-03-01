from maze.grid import Grid, SMALL
from visuals.visual_grid_example import print_static_grid

g = Grid(SMALL, seed=0)
print_static_grid(g, title="20x20 grid (seed=0)")