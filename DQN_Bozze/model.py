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
import os
import pandas as pd
import matplotlib.pyplot as plt

# hyperparameters
ENV_NAME = 'ALE/Pong-v5' # which game you want to play
MEMORY_CAPACITY = 1_000_000 #with uint8 storage
TOTAL_FRAMES = 10_000_000
START_EPSILON, END_EPSILON, STEPS = 1, 0.1, 1_000_000
AMOUNT_INPUT_FRAMES = 4
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99 # gamma
EVAL_EPSILON = 0.05
EVAL_EVERY_FRAMES = 50_000
UPDATE_TARGET_FREQ = 10_000

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
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            state_float = state.astype(np.float32) / 255.0
            state_torch = torch.from_numpy(state_float).unsqueeze(0).to(device)
            q_values = self.forward(state_torch)
            return torch.argmax(q_values, dim=1).item()

# some utils
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

        # Create seperate target network to prevent feedback loop (not described specifically enough in paper)
        self.target_network = OriginalConvNet(AMOUNT_INPUT_FRAMES, self.atari.amount_actions).to(device)
        self.target_network.load_state_dict(self.model.state_dict())
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.epsilon = Epsilon(START_EPSILON, END_EPSILON, STEPS)
        self.batch_size = BATCH_SIZE
        self.gamma = DISCOUNT_FACTOR
        self.logger = TrainLogger(self)

        if init_weights_path:
            self.model.load_state_dict(torch.load(init_weights_path))

    def q(self, batch):
        """ Get current Q values from a batch """
        state, action, reward, next_state, done = batch

        # State is already float and normalized from ReplayMemory.sample()
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        preds = self.model(state)
        used_q =  preds.gather(1, action.unsqueeze(1)).squeeze(1) # gather only q values where the action was actually taken

        return used_q


    def compute_y(self, batch):
        """ Computes the y (expected reward) needed for the bellman equation """
        state, action, reward, next_state, done = batch

        # Next_state is already float and normalized from ReplayMemory.sample()
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # run a forward pass through the model with the next state s'
        q_values_new = self.target_network(next_state)

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
        episode_num = 0
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

                    # update target network periodically
                    if frames_trained % UPDATE_TARGET_FREQ == 0:
                        self.target_network.load_state_dict(self.model.state_dict())

                # evaluate model and save every eval_every_frames frames
                if frames_trained > 0 and frames_trained % EVAL_EVERY_FRAMES == 0:
                    self.logger.eval_and_save()

                if done:
                    episode_num += 1
                    self.logger.log(episode_num, frames_trained, episode_reward)
                    break

class TrainLogger:
    def __init__(self, agent: DqnAgent):
        self.agent = agent
        self.network = self.agent.model
        self.atari = self.agent.atari
        self.eval_epsilon = EVAL_EPSILON
        self.current_best = -100.0

        # --- Logging Lists ---
        self.episode_scores = []
        self.running_avg_scores = []
        self.frame_log = []
        self.episode_log = []

        # --- Run ID and Path Setup ---
        base_dir = "models"
        os.makedirs(base_dir, exist_ok=True)

        run_id = 0
        while os.path.exists(os.path.join(base_dir, str(run_id))):
            run_id += 1

        self.run_id = run_id
        # Sanitize ENV_NAME to be a valid directory name (e.g., "ALE/Pong-v5" -> "ALE_Pong-v5")
        sanitized_env_name = ENV_NAME.replace('/', '_')
        self.save_dir = os.path.join(base_dir, str(self.run_id), sanitized_env_name)
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"Initialized Logger for Run ID: {self.run_id}")
        print(f"Models will be saved in: {self.save_dir}")

    def log(self, episode_num, frame_num, episode_reward):
        """
        Call this method after each episode to log performance.

        Args:
            episode_num (int): The current episode number.
            frame_num (int): The total number of frames trained so far.
            episode_reward (float): The total reward for the completed episode.
        """
        self.episode_scores.append(episode_reward)
        self.frame_log.append(frame_num)
        self.episode_log.append(episode_num)

        # Calculate running average over last 10 episodes
        running_avg = np.mean(self.episode_scores[-10:])
        self.running_avg_scores.append(running_avg)

        print(
            f"Ep: {episode_num} | Frames: {frame_num} | Reward: {episode_reward:.1f} | "
            f"10-Ep Avg: {running_avg:.2f} | Epsilon: {self.agent.epsilon.get():.4f}"
        )

    def _generate_reward_plot(self):
        """
        Generates and saves a plot of the training rewards.
        This is a helper method called by eval_and_save() and save().
        """
        # Avoid plotting if there's no data
        if not self.episode_log:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.episode_log, self.episode_scores, label='Episode Reward', alpha=0.5)
        plt.plot(self.episode_log, self.running_avg_scores, label='10-Episode Avg Reward', color='orange',
                 linewidth=2)
        plt.title(f'Training Rewards for {ENV_NAME} (Run ID: {self.run_id})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.save_dir, 'reward_plot.png')
        plt.savefig(plot_path)
        plt.close()  # Important to free up memory
        print(f"Updated reward plot saved to {plot_path}")

    def save(self):
        """
        Call this method at the end of training to save all statistics and plots.
        """
        print(f"\n--- Saving final run data to {self.save_dir} ---")

        # 1. Save training log data to a CSV file
        df = pd.DataFrame({
            'episode': self.episode_log,
            'frame': self.frame_log,
            'reward': self.episode_scores,
            'running_avg_reward': self.running_avg_scores
        })
        df.to_csv(os.path.join(self.save_dir, 'training_log.csv'), index=False)
        print("Training log saved to training_log.csv")

        # 2. Generate and save the final reward plot
        self._generate_reward_plot()

        # 3. Save the final model state
        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        torch.save(self.network.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

        print("--- Save complete! ---")

    def eval_and_save(self):
        """
        Evaluates the agent's performance, saves the model if it's a new best,
        and updates the reward plot for mid-training analysis.
        """
        print("\n--- Starting Evaluation ---")
        self.network.eval()  # Set the network to evaluation mode

        total_reward = 0.0
        num_episodes = 10

        for i in range(num_episodes):
            state = self.atari.reset()
            episode_reward = 0.0
            done = False

            while not done:
                # Epsilon-greedy action selection for evaluation
                if random.random() < self.eval_epsilon:
                    action = random.randrange(self.atari.amount_actions)
                else:
                    # self.network.act() already handles torch.no_grad()
                    action = self.network.act(state)

                next_state, reward, done = self.atari.step(action)

                state = next_state
                episode_reward += reward

            total_reward += episode_reward

        average_score = total_reward / num_episodes
        print(f"Evaluation Complete. Average Score over {num_episodes} episodes: {average_score:.2f}")

        if average_score > self.current_best:
            print(f"New best model found! Score: {average_score:.2f} (old best: {self.current_best:.2f})")
            self.current_best = average_score

            # Save the model state dictionary
            model_save_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(self.network.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            print(f"Score of {average_score:.2f} did not beat current best of {self.current_best:.2f}. Not saving.")

        # --- Generate and save the reward plot during training ---
        self._generate_reward_plot()

        self.network.train()  # IMPORTANT: Restore the network to training mode for the main training loop
        print("--- Finished Evaluation ---\n")

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        # We store tuples of (state, action, reward, next_state, done)
        # States are stored as uint8 to save RAM
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state and next_state are numpy arrays of dtype=uint8
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        # This is the key memory optimization.
        state = state.astype(np.float32) / 255.0
        next_state = next_state.astype(np.float32) / 255.0

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)

class AtariEnv:
    """ Wraps the env that simulates the atari game to create sufficient input for the neural net """
    def __init__(self):
        self.env = gym.make(ENV_NAME, render_mode='rgb_array', repeat_action_probability=0)
        self.amount_actions = self.env.action_space.n
        self.frames_stack = AMOUNT_INPUT_FRAMES
        # This deque will store uint8 frames
        self.frames = deque(maxlen=self.frames_stack)

    def reset(self):
        obs, _ = self.env.reset()
        processed_frame = preprocess_frames(obs)
        for _ in range(self.frames_stack):
            self.frames.append(processed_frame)
        # Return a numpy array of stacked uint8 frames
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
        # Return a numpy array of stacked uint8 frames
        return np.array(self.frames), clipped_reward, ended


def preprocess_frames(frame):
    """
    Returns a uint8 array to save memory.
    The normalization (division by 255) is deferred until a batch is sampled.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    downsampled_frame = cv2.resize(gray_frame, (84, 110))
    # Crop and keep as uint8
    cropped_frame = downsampled_frame[18:102, :]
    return cropped_frame


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    agent = DqnAgent(device)
    try:
        agent.train()
    finally:
        agent.logger.save()