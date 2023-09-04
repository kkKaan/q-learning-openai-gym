import gym
import numpy as np

# create the MountainCar environment
env = gym.make("CliffWalking-v0", render_mode="human")

# initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# set hyperparameters
num_episodes = 1000
max_steps_per_episode = 30
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# take input to check if the user wants to load the q table from a file
load_q_table = input("Do you want to load the q table from a file? (y/n): ")

if load_q_table == "n":
    # Q-learning algorithm
    for episode in range(num_episodes):
        state = env.reset()
        state = state[0]
        done = False

        for step in range(max_steps_per_episode):
            # exploration-exploitation trade-off
            exploration_rate_threshold = np.random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            # take action and observe new state and reward
            new_state, reward, done, _, _ = env.step(action)

            # update Q-table
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (
                        reward + discount_factor * np.max(q_table[new_state, :]))

            state = new_state

            if done:
                break

        # decay exploration rate
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # send the q table to a file
    np.save("q_table.npy", q_table)

else:
    # load the q table from a file
    q_table = np.load("q_table.npy")

# evaluate the agent
num_test_episodes = 10
total_rewards = []

for episode in range(num_test_episodes):
    state = env.reset()
    state = state[0]
    done = False
    episode_reward = 0

    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state, :])
        new_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        state = new_state

        if done:
            break

    total_rewards.append(episode_reward)

# close the environment
env.close()
