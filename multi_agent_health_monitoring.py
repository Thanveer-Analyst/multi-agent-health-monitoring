import numpy as np
import gym
from gym import spaces
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Hyperparameters
EPISODES = 10               # Number of training episodes
MAX_STEPS = 1000            # Max timesteps per episode
LEARNING_RATE = 0.01        # Neural network learning rate (alpha)
DISCOUNT_FACTOR = 0.9       # Discount factor for future rewards (gamma)
BATCH_SIZE = 32             # Replay memory batch size
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_DECAY = 0.995       # Decay rate per episode
EPSILON_MIN = 0.1           # Minimum exploration rate
MEMORY_SIZE = 10000         # Replay memory capacity

# MEWS thresholds for each vital sign (example values)
MEWS_THRESHOLDS = {
    'respiration': [4, 8, 20, 24, 30, 35, 36],
    'oxygen':     [84, 89, 92, 94, 95],
    'temperature':[34.0, 35.0, 36.0, 37.9, 38.5, 38.6],
    'heart_rate': [39, 49, 99, 109, 129, 139, 140]
}

# Reward policy matrix matching Table 2 in manuscript
REWARD_MATRIX = np.array([
    [-4, -3, -2, -1, 10],
    [-4, -3, -2, 10, -1],
    [-4, -3, 10, -1, -2],
    [-4, 10, -1, -2, -3],
    [10,-3, -2, -1, -4]
])

class VitalSignEnv(gym.Env):
    """
    Custom Gym environment for a single vital sign based on MEWS thresholds.
    State: scalar vital sign value. Action: discrete MET levels {0..4}.
    """
    def __init__(self, data, thresholds):
        super(VitalSignEnv, self).__init__()
        self.data = data
        self.thresholds = thresholds
        self.current_step = 0
        # Observation is a single float
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # Actions correspond to MET-0..MET-4
        self.action_space = spaces.Discrete(5)
        # Replay memory
        self.memory = deque(maxlen=MEMORY_SIZE)

    def reset(self):
        self.current_step = 0
        return np.array([self.data[self.current_step]], dtype=np.float32)

    def step(self, action):
        true_value = self.data[self.current_step]
        # Determine true MET level from thresholds
        met = np.digitize(true_value, self.thresholds) - 1
        met = np.clip(met, 0, 4)
        reward = REWARD_MATRIX[met, action]
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs = np.array([self.data[self.current_step]], dtype=np.float32) if not done else np.zeros((1,), dtype=np.float32)
        return obs, reward, done, {}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        return random.sample(self.memory, BATCH_SIZE)

# Build the DQN network
def build_model(input_shape, action_size):
    model = Sequential([
        Dense(24, activation='relu', input_shape=input_shape),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# Train a single agent
def train_agent(data, thresholds):
    env = VitalSignEnv(data, thresholds)
    model = build_model((1,), env.action_space.n)
    epsilon = EPSILON_START

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0
        for step in range(MAX_STEPS):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_vals[0])

            next_state, reward, done, _ = env.step(action)
            env.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            # Replay
            if len(env.memory) >= BATCH_SIZE:
                batch = env.sample_memory()
                states = np.array([b[0] for b in batch]).reshape(-1, 1)
                actions = np.array([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch])
                next_states = np.array([b[3] for b in batch]).reshape(-1, 1)
                dones = np.array([b[4] for b in batch])

                q_next = model.predict(next_states, verbose=0)
                q_target = rewards + DISCOUNT_FACTOR * np.max(q_next, axis=1) * (~dones)
                q_vals = model.predict(states, verbose=0)
                for i, action_i in enumerate(actions):
                    q_vals[i, action_i] = q_target[i]
                model.fit(states, q_vals, epochs=1, verbose=0)

            if done:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        print(f"Episode {ep+1}/{EPISODES} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

    return model

# Placeholder for dataset loading
# Replace these with real data loading pipelines
def load_ppg_dalia(vital):
    # Load 'vital' column data from PPG-DaLiA dataset
    return np.random.uniform(low=50, high=150, size=1000)

def load_wesad(vital):
    # Load 'vital' column data from WESAD dataset
    return np.random.uniform(low=30, high=40, size=1000)

if __name__ == "__main__":
    # Example: train heart rate agent on PPG-DaLiA
    hr_data = load_ppg_dalia('heart_rate')
    hr_thresholds = MEWS_THRESHOLDS['heart_rate']
    print("Training Heart Rate Agent...")
    hr_model = train_agent(hr_data, hr_thresholds)

    # Similarly, you can train agents for respiration and temperature
    resp_data = load_wesad('respiration')
    resp_thresholds = MEWS_THRESHOLDS['respiration']
    print("Training Respiration Agent...")
    resp_model = train_agent(resp_data, resp_thresholds)

    temp_data = load_wesad('temperature')
    temp_thresholds = MEWS_THRESHOLDS['temperature']
    print("Training Temperature Agent...")
    temp_model = train_agent(temp_data, temp_thresholds)
