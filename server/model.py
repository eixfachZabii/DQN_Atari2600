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

# Register Atari environments
gym.register_envs(ale_py)

class OriginalConvNet(nn.Module):
    """ The neural network used to predict Q values """
    def __init__(self, output_dim):
        super(Q_NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4) # output shape: 20 x 20 x 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) # output shape: 9 x 9 x 32
        self.fc = nn.Linear(32 * 9 * 9, output_dim)

    def forward(self, x):
        """
        forward pass of the neural network
        """
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DqnAgent:
    """ The main class for acting and training """
    def __init__(self, init_weights_path: str = None):
        self.atari = AtariEnv
        self.q = OriginalConvNet(self.atari.amount_actions)
        self.memory = ReplayMemory(MEMORY_CAPACITY)

        if init_weights_path:
            self.q.load_state_dict(torch.load(init_weights_path))
        pass

    def train(self):
        pass

    def act(self):
        pass

    def memorize(self):
        pass

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class AtariEnv:
    def __init__(self):
        self.env = gym.make(ENV_NAME, render_mode='rgb_array')
        self.amount_actions = self.env.action_space.n

    def get_input(self):
        """
        Get the input for the neural network as a pytorch tensor.
        """
        pass

def preprocess_frames(frame):
    """
    referred to as φ in the paper
    idea to make it better: resize to 84*84 instead of cropping such that all playing area is always visible
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    downsampled_frame = cv2.resize(gray_frame, (84, 110))
    cropped_frame = downsampled_frame[18:102, :]
    return cropped_frame

if __name__ == "__main__":
    agent = DqnAgent()
    agent.train()