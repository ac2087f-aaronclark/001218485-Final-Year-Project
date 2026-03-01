from __future__ import annotations
from maze.grid import Grid


def render_static_grid_ascii(grid: Grid) -> str:
    """
    Prints a simple ASCII view of the grid:
      # = boundary wall
      S = start
      G = goal
      1-9 = base traversal cost
    """
    lines = []
    for r in range(grid.rows):
        row_chars = []
        for c in range(grid.cols):
            if grid.walls[r, c]:
                ch = "#"
            else:
                p = (r, c)
                if p == grid.start:
                    ch = "S"
                elif p == grid.goal:
                    ch = "G"
                else:
                    ch = str(int(grid.base_cost[r, c]))  # 1..9
            row_chars.append(ch)
        lines.append("".join(row_chars))
    return "\n".join(lines)


def print_static_grid(grid: Grid, title: str | None = None) -> None:
    if title:
        print(title)
    print(render_static_grid_ascii(grid))
    print()