from __future__ import annotations

"""
This file runs the Pygame visual demo for the project.

It lets the user select a maze size, pathfinding algorithm, and dynamic mode,
then visualises the agent moving through the weighted grid while replanning
when required.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

from maze.grid import Grid, SMALL, MEDIUM, LARGE, GridSpec, Pos
from maze.dynamic_costs import (
    SpikeSystem,
    update_spikes_local_cost_spiking,
    update_spikes_path_ahead_spiking,
)
from algorithms.ucs import uniform_cost_search
from algorithms.a_star import a_star_search
from algorithms.weighted_a_star import weighted_a_star_search
from algorithms.gbfs import greedy_best_first_search
from algorithms.d_star_lite import DStarLite


# =========================
# Config
# =========================

WINDOW_W = 1280
WINDOW_H = 940
PANEL_W = 320
GRID_AREA_W = WINDOW_W - PANEL_W
FPS = 60

BACKGROUND = (20, 22, 28)
PANEL_BG = (30, 34, 42)
TEXT = (235, 235, 235)
MUTED = (170, 175, 185)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)

START_COLOUR = (60, 200, 120)
GOAL_COLOUR = (70, 140, 255)
AGENT_COLOUR = (255, 215, 70)
PATH_COLOUR = (80, 220, 255)
BUTTON_ON = (70, 110, 200)
BUTTON_OFF = (65, 70, 82)
BUTTON_BORDER = (105, 115, 135)
RUN_COLOUR = (70, 160, 90)
PAUSE_COLOUR = (190, 140, 50)
RESET_COLOUR = (170, 70, 70)

BASE_DELAY_MS = 120
SEED_BY_SIZE = {
    "Small": 7,
    "Medium": 11,
    "Large": 13,
}

W_WEIGHT = 2.0
UPDATE_EVERY_N = 10
SPIKE_WINDOW_K = 5
SPIKE_COUNT_M = 5
SPIKE_COST = 50
PATH_AHEAD_LOOKAHEAD = 10


# =========================
# UI helpers
# =========================

# Represents a clickable UI button in the side panel.
@dataclass
class Button:
    rect: pygame.Rect
    label: str
    group: str
    value: str
    colour_on: Tuple[int, int, int] = BUTTON_ON
    colour_off: Tuple[int, int, int] = BUTTON_OFF

    # Draws the button using either its selected or unselected colour.
    def draw(self, surface, font, selected: bool) -> None:
        fill = self.colour_on if selected else self.colour_off
        pygame.draw.rect(surface, fill, self.rect, border_radius=10)
        pygame.draw.rect(surface, BUTTON_BORDER, self.rect, width=2, border_radius=10)

        txt = font.render(self.label, True, WHITE)
        txt_rect = txt.get_rect(center=self.rect.center)
        surface.blit(txt, txt_rect)

    # Returns True if the given mouse position is inside the button.
    def clicked(self, pos) -> bool:
        return self.rect.collidepoint(pos)


# Creates a Button object with the supplied layout and label settings.
def make_button(x, y, w, h, label, group, value, colour_on=BUTTON_ON, colour_off=BUTTON_OFF):
    return Button(
        rect=pygame.Rect(x, y, w, h),
        label=label,
        group=group,
        value=value,
        colour_on=colour_on,
        colour_off=colour_off,
    )


# =========================
# Planning helpers
# =========================

# Runs the selected non-incremental algorithm and returns its path and nodes expanded.
def compute_path(grid: Grid, algo: str, start: Pos, goal: Pos, weight: float = W_WEIGHT):
    if algo == "UCS":
        res = uniform_cost_search(grid, start, goal)
        return res.path, int(res.nodes_expanded)
    elif algo == "A*":
        res = a_star_search(grid, start, goal)
        return res.path, int(res.nodes_expanded)
    elif algo == "wA*":
        res = weighted_a_star_search(grid, start, goal, w=weight)
        return res.path, int(res.nodes_expanded)
    elif algo == "GBFS":
        res = greedy_best_first_search(grid, start, goal)
        return res.path, int(res.nodes_expanded)
    else:
        raise ValueError(f"Unsupported non-incremental algorithm: {algo}")


# =========================
# Demo state
# =========================

# Stores the current demo configuration, runtime state, and statistics.
class DemoState:
    def __init__(self) -> None:
        self.size_name = "Small"
        self.algorithm = "A*"
        self.mode = "Baseline"

        self.grid: Optional[Grid] = None
        self.agent: Optional[Pos] = None
        self.goal: Optional[Pos] = None
        self.path: List[Pos] = []
        self.path_index = 0

        self.running = False
        self.paused = False
        self.finished = False
        self.found = False

        self.step_count = 0
        self.total_cost = 0.0
        self.replans = 0
        self.updates = 0
        self.total_nodes_expanded = 0

        self.last_tick_ms = 0
        self.delay_ms = BASE_DELAY_MS

        self.spikes: Optional[SpikeSystem] = None
        self.dstar: Optional[DStarLite] = None
        self.dstar_prev_expanded = 0

        self.status_text = "Choose settings and press Run"
        self.event_flash_frames = 0
        self.last_event_text = ""

    # Returns the GridSpec that matches the currently selected map size.
    def current_spec(self) -> GridSpec:
        if self.size_name == "Small":
            return SMALL
        if self.size_name == "Medium":
            return MEDIUM
        return LARGE

    # Resets the demo to its initial state using the current size, seed, and mode settings.
    def reset(self) -> None:
        spec = self.current_spec()
        seed = SEED_BY_SIZE[self.size_name]

        self.grid = Grid(spec, seed=seed)
        self.agent = self.grid.start
        self.goal = self.grid.goal
        self.path = []
        self.path_index = 0

        self.running = False
        self.paused = False
        self.finished = False
        self.found = False

        self.step_count = 0
        self.total_cost = 0.0
        self.replans = 0
        self.updates = 0
        self.total_nodes_expanded = 0

        self.last_tick_ms = pygame.time.get_ticks()

        self.spikes = SpikeSystem(
            spike_cost=SPIKE_COST,
            k=SPIKE_WINDOW_K,
            m=SPIKE_COUNT_M,
        )

        self.dstar = None
        self.dstar_prev_expanded = 0

        self.status_text = "Reset complete"
        self.last_event_text = ""
        self.event_flash_frames = 0

    # Starts a fresh run and creates the initial path for the selected algorithm.
    def start_run(self) -> None:
        self.reset()

        if self.algorithm == "D*Lite":
            self.dstar = DStarLite(self.grid, self.agent, self.goal)

            self.dstar_prev_expanded = self.dstar.nodes_expanded
            path = self.dstar.plan_path()
            delta = self.dstar.nodes_expanded - self.dstar_prev_expanded
            self.dstar_prev_expanded = self.dstar.nodes_expanded

            self.path = path
            self.replans += 1
            self.total_nodes_expanded += int(delta)
        else:
            path, expanded = compute_path(self.grid, self.algorithm, self.agent, self.goal)
            self.path = path
            self.replans += 1
            self.total_nodes_expanded += expanded

        self.path_index = 0
        self.running = True
        self.paused = False
        self.finished = False
        self.found = bool(self.path)

        if self.path:
            self.status_text = "Running..."
            self.flash("Initial plan created")
        else:
            self.status_text = "No path found"
            self.finished = True
            self.running = False

    # Shows a short temporary event message on the UI.
    def flash(self, text: str) -> None:
        self.last_event_text = text
        self.event_flash_frames = 22

    # Replans from the agent's current position using the selected algorithm.
    def replan_from_current_state(self) -> None:
        if self.algorithm == "D*Lite":
            self.dstar.set_start(self.agent)
            self.dstar_prev_expanded = self.dstar.nodes_expanded
            self.path = self.dstar.plan_path()
            delta = self.dstar.nodes_expanded - self.dstar_prev_expanded
            self.dstar_prev_expanded = self.dstar.nodes_expanded
            self.total_nodes_expanded += int(delta)
        else:
            self.path, expanded = compute_path(self.grid, self.algorithm, self.agent, self.goal)
            self.total_nodes_expanded += expanded

        self.replans += 1
        self.path_index = 0

    # Applies dynamic cost updates when the chosen mode and update interval require it.
    def apply_dynamic_update_if_needed(self) -> None:
        if self.mode == "Baseline":
            return

        if self.step_count == 0 or self.step_count % UPDATE_EVERY_N != 0:
            return

        if self.agent == self.goal:
            return

        self.updates += 1

        if self.mode == "Local Cost Spiking":
            changed = update_spikes_local_cost_spiking(self.grid, self.spikes, self.agent)
            self.flash("Local spikes applied")
        elif self.mode == "Path Ahead Spiking":
            changed = update_spikes_path_ahead_spiking(
                self.grid,
                self.spikes,
                self.agent,
                self.path,
                lookahead=PATH_AHEAD_LOOKAHEAD,
            )
            self.flash("Path-ahead spikes applied")
        else:
            changed = []

        if self.algorithm == "D*Lite":
            self.dstar.set_start(self.agent)
            self.dstar.notify_cost_changes(changed)

        self.replan_from_current_state()

        if not self.path:
            self.running = False
            self.finished = True
            self.found = False
            self.status_text = "No path after update"

    # Moves the agent forward by one step and triggers replanning or updates when needed.
    def advance_one_step(self) -> None:
        if not self.running or self.paused or self.finished:
            return

        if not self.path:
            self.running = False
            self.finished = True
            self.found = False
            self.status_text = "No path available"
            return

        if self.agent == self.goal:
            self.running = False
            self.finished = True
            self.found = True
            self.status_text = "Goal reached"
            return

        # If path exhausted, replan from current location.
        if self.path_index >= len(self.path) - 1:
            self.replan_from_current_state()
            if not self.path:
                self.running = False
                self.finished = True
                self.found = False
                self.status_text = "No path on replan"
                return
            self.flash("Replanned")

        next_cell = self.path[self.path_index + 1]
        self.agent = next_cell
        self.path_index += 1

        self.step_count += 1
        self.total_cost += float(self.grid.step_cost(self.agent))
        self.status_text = "Running..."

        if self.agent == self.goal:
            self.running = False
            self.finished = True
            self.found = True
            self.status_text = "Goal reached"
            self.flash("Goal reached")
            return

        self.apply_dynamic_update_if_needed()


# =========================
# Drawing
# =========================

# Computes the cell size so the grid fits inside the available drawing area.
def get_cell_size(rows: int, cols: int) -> int:
    return max(6, min(GRID_AREA_W // cols, WINDOW_H // rows))


# Draws the weighted grid, spike cells, current path, and agent markers.
def draw_grid(surface, state: DemoState, font, small_font) -> None:
    if state.grid is None:
        return

    grid = state.grid
    rows, cols = grid.rows, grid.cols
    cell = get_cell_size(rows, cols)
    offset_x = 10
    offset_y = max(10, (WINDOW_H - rows * cell) // 2)

    # Base cells
    for r in range(rows):
        for c in range(cols):
            x = offset_x + c * cell
            y = offset_y + r * cell
            rect = pygame.Rect(x, y, cell, cell)

            if grid.walls[r, c]:
                colour = (35, 35, 35)
            else:
                base = int(grid.base_cost[r, c])
                shade = 50 + int((base - 1) / 8 * 150)
                colour = (shade, shade, shade)

                if grid.spike_cost[r, c] > 0:
                    spike_strength = min(255, 130 + grid.spike_cost[r, c] * 2)
                    colour = (spike_strength, 70, 70)

            pygame.draw.rect(surface, colour, rect)

            if cell >= 16 and not grid.walls[r, c]:
                value = str(int(grid.base_cost[r, c]))
                txt = small_font.render(value, True, BLACK if sum(colour) > 300 else WHITE)
                txt_rect = txt.get_rect(center=rect.center)
                surface.blit(txt, txt_rect)

    # Current path
    if state.path:
        for i, p in enumerate(state.path):
            if p == state.agent or p == state.goal:
                continue
            r, c = p
            x = offset_x + c * cell
            y = offset_y + r * cell
            rect = pygame.Rect(x + 2, y + 2, max(2, cell - 4), max(2, cell - 4))
            pygame.draw.rect(surface, PATH_COLOUR, rect, width=2 if cell >= 10 else 1)

    # Start / goal / agent
    for p, colour, label in [
        (grid.start, START_COLOUR, "S"),
        (grid.goal, GOAL_COLOUR, "G"),
        (state.agent, AGENT_COLOUR, "A"),
    ]:
        r, c = p
        x = offset_x + c * cell
        y = offset_y + r * cell
        rect = pygame.Rect(x + 1, y + 1, max(2, cell - 2), max(2, cell - 2))
        pygame.draw.rect(surface, colour, rect, border_radius=max(2, cell // 5))

        if cell >= 18:
            txt = font.render(label, True, BLACK)
            txt_rect = txt.get_rect(center=rect.center)
            surface.blit(txt, txt_rect)

    # Grid lines
    if cell >= 10:
        for r in range(rows + 1):
            y = offset_y + r * cell
            pygame.draw.line(surface, (55, 60, 70), (offset_x, y), (offset_x + cols * cell, y), 1)
        for c in range(cols + 1):
            x = offset_x + c * cell
            pygame.draw.line(surface, (55, 60, 70), (x, offset_y), (x, offset_y + rows * cell), 1)


# Draws the fixed side panel containing settings, controls, and status information.
def draw_panel(surface, state: DemoState, title_font, font, small_font, buttons) -> None:
    panel_rect = pygame.Rect(GRID_AREA_W, 0, PANEL_W, WINDOW_H)
    pygame.draw.rect(surface, PANEL_BG, panel_rect)

    x0 = GRID_AREA_W + 18

    # Title
    y = 16
    title = title_font.render("Pathfinding Demo", True, WHITE)
    surface.blit(title, (x0, y))

    y += 40
    subtitle1 = small_font.render("Select a size, algorithm and mode,", True, MUTED)
    subtitle2 = small_font.render("then press Run.", True, MUTED)
    surface.blit(subtitle1, (x0, y))
    surface.blit(subtitle2, (x0, y + 18))

    # Map size
    surface.blit(font.render("Map Size", True, WHITE), (x0, 112))
    for b in buttons:
        if b.group == "size":
            b.draw(surface, small_font, state.size_name == b.value)

    # Algorithm
    surface.blit(font.render("Algorithm", True, WHITE), (x0, 214))
    for b in buttons:
        if b.group == "algo":
            b.draw(surface, small_font, state.algorithm == b.value)

    # Mode
    surface.blit(font.render("Mode", True, WHITE), (x0, 408))
    for b in buttons:
        if b.group == "mode":
            b.draw(surface, small_font, state.mode == b.value)

    # Colour meanings
    surface.blit(font.render("Colour Meanings", True, WHITE), (x0, 590))
    colour_items = [
        ((90, 90, 90), "Base terrain"),
        ((200, 80, 80), "Spiked cost"),
        (START_COLOUR, "Start"),
        (GOAL_COLOUR, "Goal"),
        (AGENT_COLOUR, "Agent"),
        (PATH_COLOUR, "Planned path"),
    ]

    colour_y = 620
    for colour, label in colour_items:
        pygame.draw.rect(surface, colour, pygame.Rect(x0, colour_y, 14, 14), border_radius=4)
        txt = small_font.render(label, True, WHITE)
        surface.blit(txt, (x0 + 22, colour_y - 1))
        colour_y += 18

    # Controls
    surface.blit(font.render("Controls", True, WHITE), (x0, 734))
    for b in buttons:
        if b.group == "control":
            selected = b.value == "pause" and state.paused
            b.draw(surface, small_font, selected)

    # Status
    status_y = 800
    surface.blit(font.render("Status", True, WHITE), (x0, status_y))

    status_lines = [
        f"Mode: {state.mode}",
        f"Algorithm: {state.algorithm}",
        f"Map: {state.size_name}",
        f"Step: {state.step_count}",
        f"Replans: {state.replans}",
        f"Updates: {state.updates}",
    ]

    y2 = status_y + 28
    for line in status_lines:
        txt = small_font.render(line, True, WHITE)
        surface.blit(txt, (x0, y2))
        y2 += 16

    if state.event_flash_frames > 0 and state.last_event_text:
        pulse = 180 + int(70 * math.sin(pygame.time.get_ticks() * 0.02))
        flash_colour = (pulse, pulse, 90)
        txt = font.render(state.last_event_text, True, flash_colour)
        surface.blit(txt, (x0, WINDOW_H - 32))
        state.event_flash_frames -= 1


# Builds all side-panel buttons using the current fixed UI layout.
def build_buttons():
    buttons = []

    x0 = GRID_AREA_W + 18
    bw = 84
    bh = 36
    gap = 10

    # size
    y = 144
    buttons.extend([
        make_button(x0 + 0 * (bw + gap), y, bw, bh, "Small", "size", "Small"),
        make_button(x0 + 1 * (bw + gap), y, bw, bh, "Medium", "size", "Medium"),
        make_button(x0 + 2 * (bw + gap), y, bw, bh, "Large", "size", "Large"),
    ])

    # algorithm
    y = 252
    algo_w = 124
    algo_h = 36
    buttons.extend([
        make_button(x0, y, algo_w, algo_h, "UCS", "algo", "UCS"),
        make_button(x0 + algo_w + gap, y, algo_w, algo_h, "A*", "algo", "A*"),
        make_button(x0, y + algo_h + gap, algo_w, algo_h, "Weighted A*", "algo", "wA*"),
        make_button(x0 + algo_w + gap, y + algo_h + gap, algo_w, algo_h, "GBFS", "algo", "GBFS"),
        make_button(x0, y + 2 * (algo_h + gap), 2 * algo_w + gap, algo_h, "D* Lite", "algo", "D*Lite"),
    ])

    # mode
    y = 446
    mode_h = 36
    mode_w = 184
    buttons.extend([
        make_button(x0, y, mode_w, mode_h, "Baseline", "mode", "Baseline"),
        make_button(x0, y + mode_h + gap, mode_w, mode_h, "Local Cost Spiking", "mode", "Local Cost Spiking"),
        make_button(x0, y + 2 * (mode_h + gap), mode_w, mode_h, "Path Ahead Spiking", "mode", "Path Ahead Spiking"),
    ])

    # controls
    y = 728
    buttons.extend([
        make_button(x0, y, 80, 40, "Run", "control", "run", RUN_COLOUR, RUN_COLOUR),
        make_button(x0 + 90, y, 90, 40, "Pause", "control", "pause", PAUSE_COLOUR, PAUSE_COLOUR),
        make_button(x0 + 190, y, 90, 40, "Reset", "control", "reset", RESET_COLOUR, RESET_COLOUR),
    ])

    return buttons


# =========================
# Main loop
# =========================

# Initialises Pygame, handles input, updates the demo, and draws each frame.
def main():
    pygame.init()
    pygame.display.set_caption("Pathfinding Visual Demo")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

    title_font = pygame.font.SysFont("arial", 28, bold=True)
    font = pygame.font.SysFont("arial", 17, bold=True)
    small_font = pygame.font.SysFont("arial", 14)

    buttons = build_buttons()
    state = DemoState()
    state.reset()

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and state.running and not state.finished:
                    state.paused = not state.paused
                    state.status_text = "Paused" if state.paused else "Running..."
                elif event.key == pygame.K_r:
                    state.start_run()
                elif event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                for b in buttons:
                    if not b.clicked(pos):
                        continue

                    if b.group == "size" and not state.running:
                        state.size_name = b.value
                        state.reset()

                    elif b.group == "algo" and not state.running:
                        state.algorithm = b.value
                        state.reset()

                    elif b.group == "mode" and not state.running:
                        state.mode = b.value
                        state.reset()

                    elif b.group == "control":
                        if b.value == "run":
                            state.start_run()
                        elif b.value == "pause" and state.running and not state.finished:
                            state.paused = not state.paused
                            state.status_text = "Paused" if state.paused else "Running..."
                        elif b.value == "reset":
                            state.reset()

        if state.running and not state.paused and not state.finished:
            if now - state.last_tick_ms >= state.delay_ms:
                state.advance_one_step()
                state.last_tick_ms = now

        screen.fill(BACKGROUND)
        draw_grid(screen, state, font, small_font)
        draw_panel(screen, state, title_font, font, small_font, buttons)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()