from collections import deque
from typing import List

import torch
import cv2
import numpy as np
import time
import argparse
import os
from dataclasses import dataclass
import gymnasium as gym
import ale_py
from model import preprocess_frames
from model import OriginalConvNet
from simplenet import simplenetv1_small_m1_05

# --- Configuration Constants ---
ENV_NAME = 'ALE/Pong-v5'
DIR_NAME = 'pacman_vis'
OUTPUT_RESOLUTION = (640, 840)  # Resolution for the gameplay part (Width, Height)
Q_VIS_PANEL_WIDTH = 480  # Width of the Q-value bar graph and reward graph panels
AMOUNT_INPUT_FRAMES = 4
FPS = 15
SLOWDOWN_FACTOR = 4  # How much to slow down the q-value and reward visualization videos

ACTIONS_PACMAN = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN"]
ACTIONS_PONG = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]


# --- Dataclass for Graph Styling ---
@dataclass
class LiveRewardGraphConfig:
    """Configuration for the live reward graph styling."""
    padding_top: int = 60
    padding_bottom: int = 40
    padding_x: int = 50
    graph_color: tuple = (0, 255, 0)  # Green
    axis_color: tuple = (150, 150, 150) # Gray
    zero_line_color: tuple = (0, 0, 255) # Red
    text_color: tuple = (255, 255, 255) # White
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    line_thickness: int = 2


# --- Helper Functions ---

def prepare(frames, device):
    """ Converts to numpy and normalizes frame """
    normalized = np.array(frames).astype(np.float32) / 255.0
    state_torch = torch.from_numpy(normalized).unsqueeze(0).to(device)
    return state_torch


def create_q_value_visualization(q_values: np.ndarray, actions: List[str], chosen_action_idx: int) -> np.ndarray:
    """
    Creates an image visualizing Q-values as a horizontal bar chart.
    The chosen action is highlighted with a different color and a border.
    """
    vis_height = OUTPUT_RESOLUTION[1]
    vis_width = Q_VIS_PANEL_WIDTH
    canvas = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

    # --- Normalize Q-values for bar widths ---
    q_min = q_values.min()
    q_max = q_values.max()
    q_range = q_max - q_min

    if q_range < 1e-6:
        normalized_q = np.full_like(q_values, 0.5)
    else:
        normalized_q = (q_values - q_min) / q_range

    # --- Draw Bars and Text ---
    num_actions = len(actions)
    bar_margin = 35
    bar_area_height = vis_height - (bar_margin * 2)
    bar_height = bar_area_height // num_actions
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (action, q_val, norm_q) in enumerate(zip(actions, q_values, normalized_q)):
        bar_top = bar_margin + i * bar_height
        bar_bottom = bar_top + int(bar_height * 0.8)

        color = (0, 200, 0) if i == chosen_action_idx else (255, 128, 0)

        max_bar_width = vis_width - 120
        bar_width = int(norm_q * max_bar_width)

        cv2.rectangle(canvas, (10, bar_top), (10 + bar_width, bar_bottom), color, -1)

        if i == chosen_action_idx:
            cv2.rectangle(canvas, (10, bar_top), (10 + bar_width, bar_bottom), (255, 255, 255), 2)

        text = f"{action}: {q_val:.2f}"
        cv2.putText(canvas, text, (20, bar_bottom - 10), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    title_text = "Q-Values (Per Action)"
    (text_width, text_height), _ = cv2.getTextSize(title_text, font, 0.8, 2)
    title_x = (vis_width - text_width) // 2
    cv2.putText(canvas, title_text, (title_x, bar_margin - 10), font, 0.8, (255, 255, 255), 2)

    return canvas


def create_live_reward_graph(reward_history: List[float], config: LiveRewardGraphConfig) -> np.ndarray:
    """
    Creates a live time-series graph of the maximum Q-value.
    """
    vis_height = OUTPUT_RESOLUTION[1]
    vis_width = Q_VIS_PANEL_WIDTH
    canvas = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

    # --- Title and Current Value ---
    title_text = "Live Expected Reward (Max Q)"
    (text_width, _), _ = cv2.getTextSize(title_text, config.font, 0.8, 2)
    title_x = (vis_width - text_width) // 2
    cv2.putText(canvas, title_text, (title_x, 30), config.font, 0.8, config.text_color, 2)

    if not reward_history:
        return canvas

    current_val_text = f"Current: {reward_history[-1]:.3f}"
    cv2.putText(canvas, current_val_text, (config.padding_x, config.padding_top - 5), config.font, 0.7, config.graph_color, 2)

    # --- Graph Drawing ---
    if len(reward_history) < 2:
        return canvas

    min_val = min(reward_history)
    max_val = max(reward_history)
    val_range = max_val - min_val if max_val > min_val else 1.0

    graph_area_height = vis_height - config.padding_top - config.padding_bottom
    graph_area_width = vis_width - 2 * config.padding_x

    # Draw Y-axis labels
    cv2.putText(canvas, f"{max_val:.2f}", (5, config.padding_top + 15), config.font, 0.5, config.axis_color, 1)
    cv2.putText(canvas, f"{min_val:.2f}", (5, vis_height - config.padding_bottom - 5), config.font, 0.5, config.axis_color, 1)

    # Draw zero line if it's in range
    if min_val < 0 < max_val:
        y_zero = int(config.padding_top + graph_area_height * (max_val / val_range))
        cv2.line(canvas, (config.padding_x, y_zero), (vis_width - config.padding_x, y_zero), config.zero_line_color, 1)
        cv2.putText(canvas, "0.0", (5, y_zero + 5), config.font, 0.5, config.zero_line_color, 1)


    # Generate points and draw lines
    points = []
    num_points = len(reward_history)
    for i, val in enumerate(reward_history):
        px = config.padding_x + int(i / (num_points - 1) * graph_area_width) if num_points > 1 else config.padding_x
        py = config.padding_top + int(graph_area_height * (max_val - val) / val_range)
        points.append((px, py))

    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i+1], config.graph_color, config.line_thickness)

    return canvas


def run_and_record_episodes(env, model, actions_list, gameplay_writer, q_vis_writer, live_reward_writer):
    """
    Runs a single episode, recording raw gameplay, a Q-value visualization,
    and a live reward graph to their respective video writers.
    """
    device = next(model.parameters()).device
    frame1, _ = env.reset()
    frame_queue = deque(maxlen=AMOUNT_INPUT_FRAMES)
    for _ in range(AMOUNT_INPUT_FRAMES):
        init_preprocessed = preprocess_frames(frame1)
        frame_queue.append(init_preprocessed)

    step_count = 0
    reward_history = []
    graph_config = LiveRewardGraphConfig()

    while True:
        state = prepare(frame_queue, device)

        # Infer model and get action
        with torch.no_grad():
            q_values_tensor = model(state).squeeze()

        max_q_value = q_values_tensor.max().item()
        reward_history.append(max_q_value)
        action_int = torch.argmax(q_values_tensor, dim=0).item()
        q_values_np = q_values_tensor.cpu().numpy()

        frame, reward, terminated, truncated, info = env.step(action_int)
        step_count += 1

        # 1. Prepare raw gameplay frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        resized_gameplay_frame = cv2.resize(frame_bgr, OUTPUT_RESOLUTION)

        # 2. Write to raw gameplay video
        gameplay_writer.write(resized_gameplay_frame)

        # 3. Create the Q-value bar chart panel
        q_vis_panel = create_q_value_visualization(q_values_np, actions_list, action_int)
        composite_q_vis_frame = np.hstack([resized_gameplay_frame, q_vis_panel])

        # 4. Create the live reward graph panel
        live_reward_panel = create_live_reward_graph(reward_history, graph_config)
        composite_reward_frame = np.hstack([resized_gameplay_frame, live_reward_panel])

        # 5. Write to the visualization videos (multiple times for slowdown effect)
        for _ in range(SLOWDOWN_FACTOR):
            q_vis_writer.write(composite_q_vis_frame)
            live_reward_writer.write(composite_reward_frame)

        if terminated or truncated:
            break
        frame_queue.append(preprocess_frames(frame))

    return step_count


def main():
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = gym.make(ENV_NAME, render_mode='rgb_array', repeat_action_probability=0)
    #model = simplenetv1_small_m1_05(
    #    in_chans=AMOUNT_INPUT_FRAMES,
    #    num_classes=env.action_space.n,
    #).to(device)
    model = OriginalConvNet(
        AMOUNT_INPUT_FRAMES,
        env.action_space.n,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(DIR_NAME, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # --- Setup Video Writers ---
    # 1. Raw Gameplay Writer
    gameplay_output_path = os.path.join(DIR_NAME, "raw_video.mp4")
    gameplay_writer = cv2.VideoWriter(gameplay_output_path, fourcc, FPS, OUTPUT_RESOLUTION)

    # 2. Q-Value Visualization Writer
    q_vis_output_path = os.path.join(DIR_NAME, "q_visualization.mp4")
    composite_resolution = (OUTPUT_RESOLUTION[0] + Q_VIS_PANEL_WIDTH, OUTPUT_RESOLUTION[1])
    q_vis_writer = cv2.VideoWriter(q_vis_output_path, fourcc, FPS, composite_resolution)

    # 3. Live Reward Graph Writer (NEW)
    live_reward_output_path = os.path.join(DIR_NAME, "live_reward.mp4")
    live_reward_writer = cv2.VideoWriter(live_reward_output_path, fourcc, FPS, composite_resolution)


    if not gameplay_writer.isOpened() or not q_vis_writer.isOpened() or not live_reward_writer.isOpened():
        print("Error: Could not open one or more video writers.")
        return

    print("Running episode and recording videos...")
    try:
        num_steps = run_and_record_episodes(env, model, ACTIONS_PONG, gameplay_writer, q_vis_writer, live_reward_writer)
        print(f"Episode finished after {num_steps} steps.")
    finally:
        gameplay_writer.release()
        q_vis_writer.release()
        live_reward_writer.release()
        env.close()
        print(f"Raw gameplay video saved to {gameplay_output_path}")
        print(f"Q-value visualization saved to {q_vis_output_path}")
        print(f"Live reward graph video saved to {live_reward_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained DQN agent and record videos.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model file (.pth)."
    )
    args = parser.parse_args()
    main()