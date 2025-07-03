import os
import numpy as np
import torch
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
    plt.title('loss')
    plt.plot(losses)
    print(f"Plot")
    plt.show()


def compute_loss(model, replay_buffer, batch_size, gamma, device=device):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values_old = model(state)
    q_values_new = model(next_state)

    q_value_old = q_values_old.gather(1, action.unsqueeze(1)).squeeze(1)
    q_value_new = q_values_new.max(1)[0]
    expected_q_value = reward + gamma * q_value_new * (1 - done)

    loss = (q_value_old - expected_q_value.data).pow(2).mean()

    return loss


def train(env, model, optimizer, replay_buffer, device=device):
    print("Training...")
    steps_done = 0
    episode_rewards = []
    losses = []
    model.train()

    episode = 0
    while steps_done < MAX_FRAMES:
        state, _ = env.reset()
        episode_reward = 0.0
        while True:
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
            action = model.act(state, epsilon, device)
            steps_done += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY:
                loss = compute_loss(model, replay_buffer, BATCH_SIZE, GAMMA, device)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())




            if steps_done % 10000 == 0:
                plot_stats(steps_done, episode_rewards, losses)

            if steps_done % 100_000 == 0:
                path = os.path.join(MODEL_SAVE_PATH, f"{env.spec.id}_frame_{steps_done}.pth")
                print(f"Saving weights at Frame {steps_done} ...")
                torch.save(model.state_dict(), path)

            if steps_done % 25000 == 0:
                print(f"Frame {steps_done}. Episode {episode}. Reward: {episode_rewards[-1] if episode_rewards else 0}. Loss: {np.mean(losses[-1000:]) if losses else 0}.")

            if done:
                episode_rewards.append(episode_reward)
                break

            if steps_done >= MAX_FRAMES:
                break

        episode += 1

    env.close()


def test(env, model, episodes, render=True, device=device, context=""):
    # Note: gym.wrappers.Monitor is deprecated in gymnasium
    # You may need to use alternative recording methods or remove this wrapper
    try:
        env = gym.wrappers.RecordVideo(env, VIDEO_SAVE_PATH + f'dqn_{env.spec.id}_video_{context}')
    except:
        print("Warning: Could not set up video recording. Continuing without recording.")

    model.eval()
    for episode in range(episodes):
        state, _ = env.reset()  # Updated for gymnasium API
        episode_reward = 0.0
        while True:
            action = model.act(state, 0, device)
            next_state, reward, terminated, truncated, _ = env.step(action)  # Updated for gymnasium API
            done = terminated or truncated

            if render:
                try:
                    env.render()
                    time.sleep(0.02)
                except:
                    pass  # In case rendering fails

            episode_reward += reward
            state = next_state

            if done:
                print(f"Finished Episode {episode + 1} with reward {episode_reward}")
                break

    env.close()
