import numpy as np

from maze.grid import Grid, SMALL, MEDIUM, LARGE

g1 = Grid(SMALL)
g2 = Grid(MEDIUM)
g3 = Grid(LARGE)


from maze.grid import Grid, SMALL

g = Grid(SMALL, seed=123)
print(g.base_cost[g.start], g.base_cost[g.goal])


from maze.grid import Grid, SMALL
from algorithms.UCS import uniform_cost_search

g = Grid(SMALL, seed=123)
res = uniform_cost_search(g, g.start, g.goal)

print("found path?", bool(res.path))
print("path length:", len(res.path))
print("total cost:", res.total_cost)
print("nodes expanded:", res.nodes_expanded)


from maze.grid import Grid, SMALL
from algorithms.A_hash import a_star_search

g = Grid(SMALL, seed=123)
res = a_star_search(g, g.start, g.goal)

print("found path?", bool(res.path))
print("path length:", len(res.path))
print("total cost:", res.total_cost)
print("nodes expanded:", res.nodes_expanded)


from maze.grid import Grid, SMALL
from algorithms.Weighted_A_hash import weighted_a_star_search

g = Grid(SMALL, seed=123)

w = float(input("Enter weight w (e.g., 1, 1.5, 2): ").strip())
res = weighted_a_star_search(g, g.start, g.goal, w=w)

print("w:", w)
print("found path?", bool(res.path))
print("path length:", len(res.path))
print("total cost:", res.total_cost)
print("nodes expanded:", res.nodes_expanded)


from maze.grid import Grid, SMALL
from algorithms.GBFS import greedy_best_first_search

g = Grid(SMALL, seed=123)
res = greedy_best_first_search(g, g.start, g.goal)

print("found path?", bool(res.path))
print("path length:", len(res.path))
print("total cost:", res.total_cost)
print("nodes expanded:", res.nodes_expanded)