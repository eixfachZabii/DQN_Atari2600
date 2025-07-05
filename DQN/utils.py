import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from Param import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_stats(frame_idx, rewards, losses):
    clear_output(wait=True)
    plt.figure(figsize=(20, 5))

    plt.subplot(131)
    avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else (np.mean(rewards) if rewards else 0)
    plt.title(f'Total frames {frame_idx:,} | Episodes: {len(rewards)} | Avg Reward (last 10): {avg_reward:.1f}')
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(132)
    plt.title('Loss last 10k')
    if losses:
        plt.plot(losses[-10000:])
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')

    plt.tight_layout()
    print(f"Plot")
    plt.show()


def compute_loss(current_model, target_model, replay_buffer, batch_size, gamma, device=device):
    """FIXED: Proper DQN loss with target network"""
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    # Current Q-values from main network
    q_values_current = current_model(state)
    q_value_current = q_values_current.gather(1, action.unsqueeze(1)).squeeze(1)

    # Target Q-values from target network (no gradients)
    with torch.no_grad():
        q_values_target = target_model(next_state)
        q_value_next = q_values_target.max(1)[0]
        expected_q_value = reward + gamma * q_value_next * (1 - done)

    # MSE loss between current and target Q-values
    loss = (q_value_current - expected_q_value).pow(2).mean()
    return loss


def update_target_network(current_model, target_model):
    """Copy weights from current model to target model"""
    target_model.load_state_dict(current_model.state_dict())


def train(env, current_model, target_model, optimizer, replay_buffer, device=device):
    """
    FIXED: Training loop with proper target network updates

    Key improvements:
    1. Separate current and target networks
    2. Periodic target network updates
    3. Gradient clipping for stability
    4. Better logging and monitoring
    """
    print("Starting DQN training with target network...")
    print(f"Target network update frequency: {TARGET_UPDATE_FREQ} steps")

    steps_done = 1_900_000
    episode_rewards = []
    losses = []
    current_model.train()
    target_model.eval()  # Target network always in eval mode

    episode = 0
    while steps_done < MAX_FRAMES:
        state, _ = env.reset()
        episode_reward = 0.0

        while True:
            # Epsilon-greedy exploration
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
            action = current_model.act(state, epsilon, device)
            steps_done += 1

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # Training step (only after sufficient replay buffer)
            if len(replay_buffer) > INITIAL_MEMORY:
                loss = compute_loss(current_model, target_model, replay_buffer,
                                    BATCH_SIZE, GAMMA, device)

                # Backpropagation with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=10.0)
                optimizer.step()

                losses.append(loss.item())

            # Update target network periodically
            if steps_done % TARGET_UPDATE_FREQ == 0:
                update_target_network(current_model, target_model)
                print(f"Target network updated at step {steps_done}")


            # Plotting
            if steps_done % 10000 == 0:
                plot_stats(steps_done, episode_rewards, losses)

            # Model saving
            if steps_done % 100_000 == 0:
                path = os.path.join(MODEL_SAVE_PATH, f"{env.spec.id}_frame_{steps_done}.pth")
                torch.save(current_model.state_dict(), path)
                print(f"Model saved: {path}")

            if done:
                episode_rewards.append(episode_reward)
                break

            if steps_done >= MAX_FRAMES:
                break

        episode += 1

    env.close()


def test(env, model, episodes, render=True, device=device, context=""):
    """Test function for evaluating trained models"""
    model.eval()  # Put model in evaluation mode

    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        while True:
            action = model.act(state, 0, device)  # Greedy policy (epsilon=0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if render:
                try:
                    env.render()
                    time.sleep(0.02)
                except:
                    pass

            episode_reward += reward
            state = next_state

            if done:
                print(f"Finished Episode {episode + 1} with reward {episode_reward}")
                break

    env.close()

    # Summary statistics
    if total_rewards:
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"\nTest Results ({episodes} episodes):")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Min Reward: {np.min(total_rewards):.0f}")
        print(f"Max Reward: {np.max(total_rewards):.0f}")

    return total_rewards