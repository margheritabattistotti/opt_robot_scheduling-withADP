# DYNAMIC OPTIMIZATION OF INTRALOGISTIC ROBOT SCHEDULING
## Master's Degree in Mathematical Engineering | Statistics and Optimization 
## Master's thesis in collaboration with Spindox-αHead on DARKO project.
#### Politecnico di Torino
Battistotti's resolution code for a single-robot scheduling problem through Approximate Dynamic Programming (ADP) techniques.

## Brief description
#### The problem and the objective 
Imagine a robot employed in a warehouse, where it must pick objects and transport them to designated destinations, i.e., trays.
The robot can indeed perform four action types: move in the warehouse, pick, place or throw an object. Moving and throwing risks are associated to the corresponding actions.
The objective is to find a time-efficient and risk-aware optimal scheduling of tasks. 

#### Setting
A completely connected graph is used to model the setting; its nodes are either picking or placing/throwing locations and the robot can only move along the graph's arches, which contain information on moving risk and travelling time. The graph is the result of a previously solved routing optimization problem.
#### Resolution method
The Dynamic Programming (DP) paradigm and its approximate versions are chosen as resolution methods. In particular, exact DP, Approximate Policy Iteration (API), Myopic Rollout (MR) and Monte-Carlo Tree Search (MCTS) are implemented.

## Set up
#### Environment
Pycharm

#### Data
Data are found in 
```bash 
Mio/input_data
```
 folder, where are partially hidden.
 They contain info on the problem setting: graph, nodes, arches; and info on objects and trays.

#### Files
Data are uploaded in all files where needed.
Files with "main" in their title are the main files, separated per method used. 
If to divide in sections, methods' names characterize each file and can be used as grouping rule.

## Experiments
To run experiments execute main files, named after the methods to retain information on their content.
