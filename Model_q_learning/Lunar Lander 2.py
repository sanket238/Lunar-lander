import gymnasium as gym
import numpy as np
import random
import os
# Set up Lunar Lander environment
env = gym.make("LunarLander-v2", render_mode="human")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Discretization parameters
state_buckets = 16
state_bounds = [env.observation_space.low[0], env.observation_space.high[0]]

# Q-learning parameters
learning_rate_q = 0.5
discount_factor_q = 0.95
exploration_rate_q = 1.0
exploration_decay_q = 0.995
exploration_min_q = 0.01

# Q-learning Q-table initialization
q_table = np.zeros((state_buckets, action_space))

#check if q_table exists
if os.path.exists("q_table.npy"):
    q_table = np.load("q_table.npy")
else :
    q_table = np.zeros((state_buckets, action_space))
# Discretize state
def discretize_state(state):
    scale = state_buckets / (state_bounds[1] - state_bounds[0])
    return int((state[0] - state_bounds[0]) * scale)

# Exploration-exploitation strategy for Q-learning
def get_action_q(state, exploration_rate):
    if np.random.rand() < exploration_rate:
        return env.action_space.sample()  # Explore
    return np.argmax(q_table[state, :])  # Exploit

# Main loop
for episode in range(500):
    observation = env.reset()
    state = discretize_state(observation[0])
    total_reward = 0
    for t in range(500):  # You can adjust the number of time steps per episode
        env.render()

        # Choose action based on Q-learning
        action_q = get_action_q(state, exploration_rate_q)

        # Perform action
        next_observation, reward, done, _, _ = env.step(action_q)
        total_reward += reward
        next_state = discretize_state(next_observation)

        # Q-learning update
        q_table[state, action_q] = (1 - learning_rate_q) * q_table[state, action_q] + \
                                   learning_rate_q * (reward + discount_factor_q * np.max(q_table[next_state, :]))

        state = next_state

        if done:
            break
    # save q_table after each episode
   
    print(f"Total reward for episode {episode}: {total_reward}")

    # Exploration rate decay for Q-learning
    if exploration_rate_q > exploration_min_q:
        exploration_rate_q *= exploration_decay_q
    if episode % 50 == 0:
        np.save("q_table.npy", q_table)
        print("Q-table saved at episode " + str(episode))
# save q_table after all episodes

env.close()
