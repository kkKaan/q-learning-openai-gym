# Q-Learning Agent for CliffWalking

## Project Overview
This project implements a Q-Learning agent to solve the CliffWalking environment from OpenAI Gym. The agent is trained to navigate a grid world environment, avoiding cliffs and finding the shortest path to the goal.

## Features
- Implementation of the Q-Learning algorithm.
- Epsilon-greedy strategy for action selection.
- Training and testing phases for performance evaluation.
- Ability to save and load trained Q-tables.

## Requirements
- Python 3.x
- OpenAI Gym
- NumPy

## Usage
1. Run the script: `python3 main.py`
2. Follow the prompt to load an existing Q-table or train a new agent.

## Q-Learning Agent
The agent is designed to:
- Learn optimal policies via Q-Learning.
- Use an epsilon-greedy strategy for a balance between exploration and exploitation.

## Training
- The agent is trained over a specified number of episodes, learning to maximize rewards in the CliffWalking environment.
- The Q-table records the value of taking certain actions in specific states.

## Testing
- The agent's performance is evaluated over a number of test episodes.
- Rewards per episode are recorded to gauge the effectiveness of the learned policy.
