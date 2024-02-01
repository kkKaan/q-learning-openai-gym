"""
This file contains a simple implementation of Q-learning algorithm on an environment 
with discrete action and observation spaces.
"""

import gym
import numpy as np

class QLearningAgent:
    def __init__(self, env_name, num_episodes, max_steps_per_episode):
        # Initialize the gym environment, state and action spaces
        self.env = gym.make(env_name)
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n

        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.num_states, self.num_actions))

        # Set training parameters
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = 0.1
        self.discount_factor = 0.99

        # Exploration parameters for epsilon-greedy strategy
        self.exploration_rate = 1.0
        self.max_exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.01

    def choose_action(self, state):
        # Decide whether to explore or exploit based on exploration rate
        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:
            return np.argmax(self.q_table[state, :])  # Exploitation: choose best known action
        else:
            return self.env.action_space.sample()  # Exploration: choose random action

    def train(self):
        # Training loop
        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = state[0]
            done = False

            for step in range(self.max_steps_per_episode):
                action = self.choose_action(state)
                new_state, reward, done, _, _ = self.env.step(action)

                # Update Q-table using the Q-learning algorithm
                self.q_table[state, action] = self.q_table[state, action] * (1 - self.learning_rate) + \
                                              self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[new_state, :]))
                state = new_state

                if done:
                    break

            # Update exploration rate
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * \
                                        np.exp(-self.exploration_decay_rate * episode))

    def save_q_table(self, filename):
        # Save the Q-table to a file
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        # Load the Q-table from a file
        self.q_table = np.load(filename)

    def test(self, num_test_episodes):
        # Testing loop
        total_rewards = []

        for episode in range(num_test_episodes):
            state = self.env.reset()
            state = state[0]
            done = False
            episode_reward = 0

            for step in range(self.max_steps_per_episode):
                action = np.argmax(self.q_table[state, :])  # Exploitation: choose best known action
                new_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = new_state

                if done:
                    break

            total_rewards.append(episode_reward)

        self.env.close()
        return total_rewards

def main():
    env_name = "CliffWalking-v0"
    num_episodes = 1000
    max_steps_per_episode = 30
    num_test_episodes = 10

    agent = QLearningAgent(env_name, num_episodes, max_steps_per_episode)

    # Prompt to load an existing Q-table
    load_q_table = input("Do you want to load the q table from a file? (y/n): ")

    if load_q_table == "n":
        agent.train()
        agent.save_q_table("q_table.npy")
    else:
        agent.load_q_table("q_table.npy")

    test_rewards = agent.test(num_test_episodes)
    print("Average test reward:", np.mean(test_rewards))

if __name__ == "__main__":
    main()
