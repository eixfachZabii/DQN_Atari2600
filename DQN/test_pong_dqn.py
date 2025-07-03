import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
import time
import argparse
from collections import deque
import imageio
import ale_py
import os

# Register ALE environments
gym.register_envs(ale_py)


# Import preprocessing wrappers (same as training)
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0

        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
                break
        return obs, info

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=new_shape,
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def make_atari_env(env_id):
    """Create Atari environment with same preprocessing as training"""
    env = gym.make(env_id, render_mode='rgb_array')
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = ImageToPyTorch(env)
    return env


# DQN Model (same architecture as training)
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def act(self, state, epsilon, device):
        if np.random.random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = np.random.randint(self.num_actions)
        return action


class PongTester:
    def __init__(self, model_path=None, device='auto'):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'auto' else "cpu")
        print(f"Using device: {self.device}")

        # Setup environment
        self.env = make_atari_env('ALE/Pong-v5')

        # Get action meanings for Pong
        temp_env = gym.make('ALE/Pong-v5')
        self.action_meanings = temp_env.unwrapped.get_action_meanings()
        temp_env.close()

        # Load trained model
        self.state_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.model = DQN(self.state_shape, self.n_actions).to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"Successfully loaded model from {model_path}")
                print(f"Model architecture: {self.state_shape} -> {self.n_actions} actions")
                print(f"Actions: {self.action_meanings}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random agent instead.")
                self.model = None
        else:
            print(f"Model file not found: {model_path}. Using random agent.")
            self.model = None

    def get_action(self, state, epsilon=0.0):
        """Get action from trained model"""
        if self.model is None:
            return self.env.action_space.sample()

        return self.model.act(state, epsilon, self.device)

    def play_episode(self, render=True, save_video=False, video_path="pong_gameplay.gif"):
        """Play a single episode and optionally save video"""
        state, _ = self.env.reset()
        total_reward = 0
        steps = 0
        frames = []

        print("Starting Pong episode...")

        while True:
            # Get action from model
            action = self.get_action(state, epsilon=0.0)  # Greedy policy

            # Take action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Capture frame for video/rendering
            if save_video or render:
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)

            # Show live gameplay
            if render and steps % 2 == 0:  # Show every 2nd frame to reduce flicker
                if frame is not None:
                    cv2.imshow('Pong DQN Agent', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Interrupted by user")
                        break

            state = next_state

            if done:
                break

        if render:
            cv2.destroyAllWindows()

        # Save video if requested
        if save_video and frames:
            print(f"Saving video to {video_path}...")
            # Reduce frame rate for smaller file size
            frames_reduced = frames[::4]  # Take every 4th frame
            imageio.mimsave(video_path, frames_reduced, fps=15)
            print(f"Video saved with {len(frames_reduced)} frames")

        return total_reward, steps, len(frames)

    def evaluate_agent(self, num_episodes=5, save_videos=False):
        """Evaluate agent over multiple episodes"""
        scores = []
        episode_lengths = []

        print(f"\nEvaluating Pong agent over {num_episodes} episodes...")
        print("-" * 60)

        for episode in range(num_episodes):
            video_path = f"pong_episode_{episode + 1}.gif" if save_videos else None
            score, steps, frames = self.play_episode(
                render=False,
                save_video=save_videos,
                video_path=video_path
            )

            scores.append(score)
            episode_lengths.append(steps)

            print(f"Episode {episode + 1:2d}: Score = {score:6.0f}, Steps = {steps:4d}")

        # Statistics
        print("-" * 60)
        print(f"Average Score: {np.mean(scores):6.2f} ± {np.std(scores):.2f}")
        print(f"Best Score:    {np.max(scores):6.0f}")
        print(f"Worst Score:   {np.min(scores):6.0f}")
        print(f"Avg Steps:     {np.mean(episode_lengths):6.1f}")

        # Pong-specific analysis
        wins = sum(1 for score in scores if score > 0)
        losses = sum(1 for score in scores if score < 0)
        draws = sum(1 for score in scores if score == 0)

        print(f"\nPong Results:")
        print(f"Wins:   {wins:2d} ({wins / num_episodes * 100:4.1f}%)")
        print(f"Losses: {losses:2d} ({losses / num_episodes * 100:4.1f}%)")
        print(f"Draws:  {draws:2d} ({draws / num_episodes * 100:4.1f}%)")

        return scores, episode_lengths

    def plot_q_values(self, state):
        """Visualize Q-values for current state"""
        if self.model is None:
            print("No model loaded - cannot show Q-values")
            return

        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()[0]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(q_values)), q_values)
        plt.title("Q-Values for Current State (Pong)")
        plt.xlabel("Actions")
        plt.ylabel("Q-Value")
        plt.xticks(range(len(q_values)), self.action_meanings, rotation=45)

        # Highlight best action
        best_action = np.argmax(q_values)
        bars[best_action].set_color('red')

        plt.tight_layout()
        plt.show()

        print(f"Best action: {self.action_meanings[best_action]} (Q-value: {q_values[best_action]:.3f})")

        # Show all Q-values
        for i, (action, q_val) in enumerate(zip(self.action_meanings, q_values)):
            marker = " <-- BEST" if i == best_action else ""
            print(f"  {action:12s}: {q_val:7.3f}{marker}")

    def interactive_demo(self):
        """Interactive demo with real-time Q-value display"""
        print("\nStarting interactive Pong demo...")
        print("Controls:")
        print("  'q' - quit")
        print("  's' - show Q-values")
        print("  'p' - pause/unpause")
        print("  'r' - reset game")

        state, _ = self.env.reset()
        total_reward = 0
        paused = False
        step_count = 0

        while True:
            if not paused:
                action = self.get_action(state, epsilon=0.0)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1

                action_name = self.action_meanings[action] if action < len(self.action_meanings) else str(action)
                print(f"\rStep: {step_count:4d} | Score: {total_reward:6.0f} | Action: {action_name:12s}", end="")

                state = next_state

                if done:
                    print(f"\nGame Over! Final Score: {total_reward} in {step_count} steps")
                    print("Press 'r' to restart or 'q' to quit")
                    paused = True

            # Render frame
            frame = self.env.render()
            if frame is not None:
                cv2.imshow('Pong DQN Agent (Interactive)', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Handle keyboard input
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\nShowing Q-values...")
                self.plot_q_values(state)
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
            elif key == ord('r'):
                print("\nRestarting game...")
                state, _ = self.env.reset()
                total_reward = 0
                step_count = 0
                paused = False

        cv2.destroyAllWindows()

    def benchmark_model(self, num_episodes=100):
        """Run extensive evaluation for model benchmarking"""
        print(f"Running benchmark with {num_episodes} episodes...")

        scores = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0

            while True:
                action = self.get_action(state, epsilon=0.0)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state

                if done:
                    break

            scores.append(total_reward)
            if (episode + 1) % 10 == 0:
                avg_score = np.mean(scores[-10:])
                print(f"Episodes {episode - 8:3d}-{episode + 1:3d}: Avg Score = {avg_score:6.2f}")

        # Final statistics
        print(f"\nBenchmark Results ({num_episodes} episodes):")
        print(f"Mean Score: {np.mean(scores):6.2f}")
        print(f"Std Score:  {np.std(scores):6.2f}")
        print(f"Min Score:  {np.min(scores):6.0f}")
        print(f"Max Score:  {np.max(scores):6.0f}")

        # Win rate
        wins = sum(1 for score in scores if score > 0)
        win_rate = wins / num_episodes * 100
        print(f"Win Rate:   {win_rate:5.1f}% ({wins}/{num_episodes})")

        return scores


def main():
    parser = argparse.ArgumentParser(description='Test DQN Pong Agent')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to saved model (.pth file)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--save-videos', action='store_true',
                        help='Save gameplay videos as GIFs')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive demo')
    parser.add_argument('--single-game', action='store_true',
                        help='Play single game with live rendering')
    parser.add_argument('--benchmark', type=int, default=0,
                        help='Run benchmark with specified number of episodes')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for inference')

    args = parser.parse_args()

    # Create tester
    tester = PongTester(args.model, device=args.device)

    if args.benchmark > 0:
        scores = tester.benchmark_model(args.benchmark)

        # Plot benchmark results
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(scores, alpha=0.7)
        plt.title(f'Scores over {args.benchmark} Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.subplot(1, 3, 2)
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        plt.subplot(1, 3, 3)
        # Rolling average
        window = min(20, len(scores) // 5)
        if window > 1:
            rolling_avg = [np.mean(scores[max(0, i - window):i + 1]) for i in range(len(scores))]
            plt.plot(rolling_avg)
            plt.title(f'Rolling Average (window={window})')
            plt.xlabel('Episode')
            plt.ylabel('Average Score')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    elif args.interactive:
        tester.interactive_demo()
    elif args.single_game:
        print("Playing single game with live rendering...")
        score, steps, frames = tester.play_episode(render=True, save_video=args.save_videos)
        print(f"\nGame finished! Score: {score}, Steps: {steps}")
    else:
        # Standard evaluation
        scores, lengths = tester.evaluate_agent(args.episodes, args.save_videos)

        # Plot results
        if len(scores) > 1:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(scores, 'bo-', alpha=0.7)
            plt.title('Scores per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Draw line')
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.hist(scores, bins=min(10, len(scores)), alpha=0.7, edgecolor='black')
            plt.title('Score Distribution')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Draw line')
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.bar(['Wins', 'Losses', 'Draws'],
                    [sum(1 for s in scores if s > 0),
                     sum(1 for s in scores if s < 0),
                     sum(1 for s in scores if s == 0)],
                    color=['green', 'red', 'gray'], alpha=0.7)
            plt.title('Game Outcomes')
            plt.ylabel('Count')

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()