import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import argparse
from collections import deque
import imageio
import ale_py
import os

from wrappers import *
from model import *

# Register ALE environments
gym.register_envs(ale_py)


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

        # Create action grouping
        self.grouped_actions = ['NOOP', 'UP', 'DOWN']
        self.action_groups = self._create_action_groups()

        # Load trained model
        self.state_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.model = DQN(self.state_shape, self.n_actions).to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.eval()
                print(f"Successfully loaded model from {model_path}")
                print(f"Model architecture: {self.state_shape} -> {self.n_actions} actions")
                print(f"Original actions: {self.action_meanings}")
                print(f"Grouped actions: {self.grouped_actions}")
                print(f"Action grouping: {self.action_groups}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random agent instead.")
                self.model = None
        else:
            print(f"Model file not found: {model_path}. Using random agent.")
            self.model = None

    def _create_action_groups(self):
        """Create mapping from original actions to grouped actions"""
        action_groups = {'NOOP': [], 'UP': [], 'DOWN': []}

        for i, action_name in enumerate(self.action_meanings):
            action_upper = action_name.upper()

            # Group NOOP and FIRE together
            if 'NOOP' in action_upper or action_upper == 'FIRE':
                action_groups['NOOP'].append(i)
            # Group UP/RIGHT movements (and their FIRE variants)
            elif 'UP' in action_upper or 'RIGHT' in action_upper:
                action_groups['UP'].append(i)
            # Group DOWN/LEFT movements (and their FIRE variants)
            elif 'DOWN' in action_upper or 'LEFT' in action_upper:
                action_groups['DOWN'].append(i)
            else:
                # Default case - put unrecognized actions in NOOP
                action_groups['NOOP'].append(i)

        return action_groups

    def _get_grouped_q_values(self, q_values):
        """Convert individual Q-values to grouped Q-values"""
        grouped_q_values = {}
        grouped_best_actions = {}

        for group_name, action_indices in self.action_groups.items():
            if action_indices:
                # Take the maximum Q-value within each group
                group_q_values = [q_values[i] for i in action_indices]
                max_q_value = max(group_q_values)
                best_action_idx = action_indices[np.argmax(group_q_values)]

                grouped_q_values[group_name] = max_q_value
                grouped_best_actions[group_name] = (best_action_idx, self.action_meanings[best_action_idx])
            else:
                grouped_q_values[group_name] = 0.0
                grouped_best_actions[group_name] = (0, 'NONE')

        return grouped_q_values, grouped_best_actions

    def get_action(self, state, epsilon=0.0):
        """Get action from trained model"""
        if self.model is None:
            return self.env.action_space.sample()

        return self.model.act(state, epsilon, self.device)

    def get_q_values(self, state):
        """Get Q-values for current state"""
        if self.model is None:
            return np.zeros(self.n_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()[0]
        return q_values

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
        """Visualize grouped Q-values for current state"""
        if self.model is None:
            print("No model loaded - cannot show Q-values")
            return

        # Get original Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()[0]

        # Convert to grouped Q-values
        grouped_q_values, grouped_best_actions = self._get_grouped_q_values(q_values)

        plt.figure(figsize=(12, 8))

        # Plot grouped Q-values
        plt.subplot(2, 1, 1)
        group_names = list(grouped_q_values.keys())
        group_q_vals = list(grouped_q_values.values())
        bars = plt.bar(range(len(group_q_vals)), group_q_vals)
        plt.title("Grouped Q-Values for Current State (Pong)")
        plt.xlabel("Action Groups")
        plt.ylabel("Q-Value")
        plt.xticks(range(len(group_names)), group_names)

        # Highlight best group
        best_group_idx = np.argmax(group_q_vals)
        bars[best_group_idx].set_color('red')

        # Plot original Q-values for reference
        plt.subplot(2, 1, 2)
        bars_orig = plt.bar(range(len(q_values)), q_values, alpha=0.7)
        plt.title("Original Q-Values (All Actions)")
        plt.xlabel("Actions")
        plt.ylabel("Q-Value")
        plt.xticks(range(len(q_values)), self.action_meanings, rotation=45)

        # Highlight best original action
        best_action_orig = np.argmax(q_values)
        bars_orig[best_action_orig].set_color('red')

        plt.tight_layout()
        plt.show()

        print(f"\nGrouped Q-values:")
        best_group = group_names[best_group_idx]
        for group_name in group_names:
            q_val = grouped_q_values[group_name]
            best_action_idx, best_action_name = grouped_best_actions[group_name]
            marker = " <-- BEST GROUP" if group_name == best_group else ""
            print(f"  {group_name:8s}: {q_val:7.3f} (best: {best_action_name}){marker}")

        print(
            f"\nOverall best action: {self.action_meanings[best_action_orig]} (Q-value: {q_values[best_action_orig]:.3f})")

    def interactive_demo(self):
        """Interactive demo with real-time grouped Q-value display"""
        print("\nStarting interactive Pong demo with live grouped Q-values...")
        print("Controls (click on the matplotlib window first, then use keys):")
        print("  'q' - quit")
        print("  'p' - pause/unpause")
        print("  'r' - reset game")
        print("  '+' or '=' - increase speed")
        print("  '-' - decrease speed")
        print("  '0' - reset to normal speed")
        print("  '1'-'9' - set speed presets (1=slowest, 9=fastest)")
        print("Close the matplotlib window to exit.")

        # Speed control variables
        base_delay = 0.05  # Base delay in seconds
        speed_presets = {
            '1': 0.2,  # Very slow
            '2': 0.15,  # Slow
            '3': 0.1,  # Slower
            '4': 0.08,  # Slightly slow
            '5': 0.05,  # Normal
            '6': 0.03,  # Slightly fast
            '7': 0.02,  # Fast
            '8': 0.01,  # Very fast
            '9': 0.005  # Ultra fast
        }

        # Game state variables
        state, _ = self.env.reset()
        total_reward = 0
        paused = False
        step_count = 0
        current_delay = base_delay
        should_quit = False

        def get_speed_description(delay):
            fps = 1 / delay
            if fps < 5:
                return "Very Slow"
            elif fps < 10:
                return "Slow"
            elif fps < 15:
                return "Normal"
            elif fps < 30:
                return "Fast"
            elif fps < 50:
                return "Very Fast"
            else:
                return "Ultra Fast"

        def on_key_press(event):
            nonlocal paused, state, total_reward, step_count, current_delay, should_quit

            if event.key == 'q':
                should_quit = True
                print("\nQuitting...")
            elif event.key == 'p':
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
            elif event.key == 'r':
                print("\nRestarting game...")
                state, _ = self.env.reset()
                total_reward = 0
                step_count = 0
                paused = False
            elif event.key in ['+', '=']:
                current_delay = max(0.001, current_delay * 0.8)  # Increase speed
                speed_desc = get_speed_description(current_delay)
                print(f"\nSpeed increased: {1 / current_delay:.1f} FPS ({speed_desc})")
            elif event.key == '-':
                current_delay = min(1.0, current_delay * 1.25)  # Decrease speed
                speed_desc = get_speed_description(current_delay)
                print(f"\nSpeed decreased: {1 / current_delay:.1f} FPS ({speed_desc})")
            elif event.key == '0':
                current_delay = base_delay  # Reset to normal speed
                speed_desc = get_speed_description(current_delay)
                print(f"\nSpeed reset to normal: {1 / current_delay:.1f} FPS ({speed_desc})")
            elif event.key in speed_presets:
                current_delay = speed_presets[event.key]
                speed_desc = get_speed_description(current_delay)
                print(f"\nSpeed preset {event.key}: {1 / current_delay:.1f} FPS ({speed_desc})")

        # Set up matplotlib for live plotting
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Connect the key press event
        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Initialize grouped Q-values plot
        ax2.set_title("Live Grouped Q-Values")
        ax2.set_xlabel("Action Groups")
        ax2.set_ylabel("Q-Value")
        ax2.set_xticks(range(len(self.grouped_actions)))
        ax2.set_xticklabels(self.grouped_actions)

        # Initialize game frame plot
        ax1.set_title("Game Frame")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Initial empty plots
        q_bars = ax2.bar(range(len(self.grouped_actions)), [0] * len(self.grouped_actions), alpha=0.7)
        game_img = ax1.imshow(np.zeros((210, 160, 3), dtype=np.uint8))

        plt.tight_layout()
        plt.show()

        print(f"Initial speed: {1 / current_delay:.1f} FPS ({get_speed_description(current_delay)})")
        print("Make sure to click on the matplotlib window to activate keyboard controls!")

        while plt.fignum_exists(fig.number) and not should_quit:  # Continue while figure is open
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
                    print("Game will restart automatically in 2 seconds...")
                    plt.pause(2)
                    state, _ = self.env.reset()
                    total_reward = 0
                    step_count = 0

            # Render frame and update plots
            frame = self.env.render()
            if frame is not None:
                # Update game frame
                game_img.set_array(frame)

                # Update grouped Q-values
                if self.model is not None:
                    q_values = self.get_q_values(state)
                    grouped_q_values, grouped_best_actions = self._get_grouped_q_values(q_values)

                    # Update bar heights
                    group_names = list(grouped_q_values.keys())
                    group_q_vals = list(grouped_q_values.values())

                    for bar, q_val in zip(q_bars, group_q_vals):
                        bar.set_height(q_val)

                    # Color bars (best group in red, others in blue)
                    best_group_idx = np.argmax(group_q_vals)
                    for i, bar in enumerate(q_bars):
                        if i == best_group_idx:
                            bar.set_color('red')
                        else:
                            bar.set_color('blue')

                    # Update y-axis limits to fit data
                    if group_q_vals:
                        y_min, y_max = min(group_q_vals), max(group_q_vals)
                        y_range = y_max - y_min
                        if y_range > 0:
                            ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

                        # Update title with current best group
                        best_group = group_names[best_group_idx]
                        best_action_idx, best_action_name = grouped_best_actions[best_group]
                        ax2.set_title(
                            f"Live Grouped Q-Values | Best: {best_group} ({group_q_vals[best_group_idx]:.3f})\n"
                            f"Actual action: {best_action_name}")

                # Update info in game frame title
                ax1.set_title(
                    f"Game Frame | Step: {step_count} | Score: {total_reward} | Speed: {1 / current_delay:.1f} FPS")

            # Update the plot with configurable delay
            plt.pause(max(0.001, current_delay))  # Use current delay for speed control

        plt.ioff()  # Turn off interactive mode
        plt.close(fig)

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