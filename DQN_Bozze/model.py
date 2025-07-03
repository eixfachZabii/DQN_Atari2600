import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
import gymnasium as gym
import ale_py
import time

# hyperparameters
ENV_NAME = 'ALE/Pong-v5' # which game you want to play
MEMORY_CAPACITY = 1e6 # paper mentions a replay memory capacity of 1 million
TOTAL_FRAMES = 10e6
START_EPSILON, END_EPSILON, STEPS = 1, 0.1, 1e6
AMOUNT_INPUT_FRAMES = 4
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99 # gamma

# Register Atari environments
gym.register_envs(ale_py)

class OriginalConvNet(nn.Module):
    """ The neural network used to predict Q values """
    def __init__(self, input_frames, output_dim):
        """ output_dim corresponds to the number of possible actions """
        super().__init__()

        # Convolutional Layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_frames, 16, kernel_size=8, stride=4),  # output shape: 20 x 20 x 16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # output shape: 9 x 9 x 32
            nn.ReLU(),
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        """
        forward pass of the neural network
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def act(self, state):
        """ get action a_t given a state (4 stacked frames) """
        self.eval()
        with torch.no_grad():
            state_torch = torch.from_numpy(state).unsqueeze(0).float().to(device)
            q_values = self.forward(state_torch)
            return torch.argmax(q_values, dim=1).item()

class Epsilon:
    def __init__(self, start_value, end_value, decrement_until):
        self.decrement_until = decrement_until
        self.decr = (start_value - end_value) / decrement_until
        self.curr_step = 0
        self.epsilon = start_value

    def get_and_update(self):
        self.curr_step += 1
        current = self.epsilon
        if self.curr_step <= self.decrement_until:
            self.epsilon -= self.decr
        return current

    def get(self):
        return self.epsilon


class DqnAgent:
    """ The main class for acting and training """
    def __init__(self, device, init_weights_path: str = None):
        self.device = device
        self.atari = AtariEnv()
        self.model = OriginalConvNet(AMOUNT_INPUT_FRAMES, self.atari.amount_actions).to(device)
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.epsilon = Epsilon(START_EPSILON, END_EPSILON, STEPS)
        self.batch_size = BATCH_SIZE
        self.gamma = DISCOUNT_FACTOR

        if init_weights_path:
            self.model.load_state_dict(torch.load(init_weights_path))

    def q(self, batch):
        """ Get current Q values from a batch """
        state, action, reward, next_state, done = batch

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        preds = self.model(state)
        used_q =  preds.gather(1, action.unsqueeze(1)).squeeze(1) # gather only q values where the action was actually taken

        return used_q


    def compute_y(self, batch):
        """ Computes the y (expected reward) needed for the bellman equation """
        state, action, reward, next_state, done = batch

        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # run a forward pass through the model with the next state s'
        q_values_new = self.model(next_state)

        # select the highest q value
        q_value_new = q_values_new.max(1)[0]

        # r_j + γ * max_a′ Q(φj+1, a′; θ)
        y = reward + self.gamma * q_value_new * (1 - done)

        return y

    def gradient_descent(self, loss):
        """ Performs a gradient descent step given the computed loss """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        frames_trained = 0
        while frames_trained < TOTAL_FRAMES: # for episode = 1, M do
            state = self.atari.reset()
            episode_reward = 0.
            while True: # for t = 1, T do
                frames_trained += 1

                # With probability epsilon select a random action at
                if random.random() < self.epsilon.get_and_update():
                    action = random.randrange(self.atari.amount_actions)
                else:
                    action = self.model.act(state)

                # Execute action a_t in emulator and observe reward rt and image xt+1
                next_state, reward, done = self.atari.step(action)

                # Store transition (φt, at, rt, φt+1) in D (memory)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(self.memory) > BATCH_SIZE:
                    # Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
                    batch = self.memory.sample(self.batch_size)

                    # Compute Loss
                    q_values = self.q(batch)
                    y = self.compute_y(batch)
                    loss = (y - q_values).pow(2).mean()   # mean squared error on actual vs predicted reward

                    # Perform gradient descent step
                    self.gradient_descent(loss)

                if done:
                    break

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state.astype(np.float32), action, reward, next_state.astype(np.float32), done

    def __len__(self):
        return len(self.memory)

class AtariEnv:
    """ Wraps the env that simulates the atari game to create sufficient input for the neural net """
    def __init__(self):
        self.env = gym.make(ENV_NAME, render_mode='rgb_array')
        self.amount_actions = self.env.action_space.n
        self.frames_stack = AMOUNT_INPUT_FRAMES
        self.frames = deque(maxlen=self.frames_stack)

    def reset(self):
        obs, _ = self.env.reset()
        processed_frame = preprocess_frames(obs)
        for _ in range(self.frames_stack):
            self.frames.append(processed_frame)
        return np.array(self.frames)

    def step(self, action):
        full_reward = 0.
        ended = False
        for _ in range(self.frames_stack):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(preprocess_frames(obs))
            full_reward += reward
            ended = terminated or truncated
            if ended:
                break

        # clip rewards to be 1 for positive rewards, -1 for negative and 0 for nothing
        clipped_reward = np.sign(full_reward)
        return np.array(self.frames), clipped_reward, ended


def preprocess_frames(frame):
    """
    referred to as φ in the paper
    idea to make it better: resize to 84*84 instead of cropping such that all playing area is always visible
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    downsampled_frame = cv2.resize(gray_frame, (84, 110))
    cropped_frame = downsampled_frame[18:102, :] / 255 # divide by 255 to normalize between 0 and 1
    return cropped_frame

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DqnAgent(device)
    agent.train()