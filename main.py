import gym
import numpy as np

class QLearningAgent:
    def __init__(self, env_name, num_episodes, max_steps_per_episode):
        self.env = gym.make(env_name)
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 1.0
        self.max_exploration_rate = 1.0
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.01

    def choose_action(self, state):
        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:
            return np.argmax(self.q_table[state, :])
        else:
            return self.env.action_space.sample()

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = state[0]
            done = False

            for step in range(self.max_steps_per_episode):
                action = self.choose_action(state)
                new_state, reward, done, _, _ = self.env.step(action)
                self.q_table[state, action] = self.q_table[state, action] * (1 - self.learning_rate) + self.learning_rate * (
                            reward + self.discount_factor * np.max(self.q_table[new_state, :]))

                state = new_state

                if done:
                    break

            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * np.exp(-self.exploration_decay_rate * episode))

    def save_q_table(self, filename):
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        self.q_table = np.load(filename)

    def test(self, num_test_episodes):
        total_rewards = []

        for episode in range(num_test_episodes):
            state = self.env.reset()
            state = state[0]
            done = False
            episode_reward = 0

            for step in range(self.max_steps_per_episode):
                action = np.argmax(self.q_table[state, :])
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
