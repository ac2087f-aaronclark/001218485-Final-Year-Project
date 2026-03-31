Name: Aaron Clark  Student ID: 001218485  Supervisor: Rafael Martinez Torres  

This repository contains the coding component of my Final Year Project - Investigating Pathfinding Algorotihms in weighted grid mazes with dynamic obstacles.

Overview:
This project compares several pathfinding algorithms on weighted grid mazes with dynamic cost spikes.  
The algorithms are tested in both a baseline static control setting and dynamic settings where one of 2 conidtions affect traversal costs during execution.  
A Pygame visual demo is also included to demonstrate the algorithms operating on the grid.

Algorithms:
Uniform Cost Search (UCS)
A*
Weighted A*
Greedy Best-First Search (GBFS)
D* Lite

Project Structure:

algorithms folder: Contains the implementations of the pathfinding algorithms used in the project.

maze folder: Contains the grid.py and dynamic_cost.py 
grid.py: defines the weighted grid, boundary walls, start/goal positions, neighbours, and traversal costs
dynamic_costs.py: applies and clears dynamic cost spikes for the two dynamic update rules

experiments folder: responsbble for logic regarding running the algorithms and measures metrics
baseline_experiment.py: runs baseline static control experiments
dynamic_experiment.py: runs dynamic experiments with replanning and cost updates

main folder: contains the main scripts used to run the experiments and create result files.
run_baseline.py: runs the baseline control experiment batch
run_dynamic.py: runs both dynamic experiment modes 

visual_demo: contains the Pygame visualisation.
visual_demo_main.py: runs and launches the interactive demo where the user can choose map size, algorithm, and dynamic mode

results folder: stores the CSV output files and summary files produced by the experiment scripts.


Setup and Execution:
Install the required python packages: numpy pandas pygame
To run the experiemnts and get results run the 2 files under the main folder
To run the visual demo and run the file under the visual_demo folder
