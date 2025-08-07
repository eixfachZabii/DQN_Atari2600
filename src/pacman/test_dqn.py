import torch
import cv2
import numpy as np
import time
import argparse
import os
from simplenet import simplenetv1_small_m1_05

# --- Import necessary classes and constants from your model.py file ---
# Make sure model.py is in the same directory as this script.
from model import OriginalConvNet, AtariEnv, AMOUNT_INPUT_FRAMES, ENV_NAME

# --- Constants for the test script ---
EVAL_EPSILON = 0.05  # A small epsilon for some exploratory actions during testing
VIDEO_FILENAME = "pong_simulation.mp4"
FRAME_RATE = 25 # Frame rate for the output video

def test_agent(model_path: str):
    """
    Loads a trained model and runs a visual simulation of the agent playing Pong.

    Args:
        model_path (str): The path to the saved .pth model file.
    """
    # --- 1. Setup Device and Environment ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the Atari environment wrapper from model.py
    # We use the 'human' render mode to get frames for visualization
    env = AtariEnv()
    # We need the original gym env to get the rendered frames
    raw_env = env.env

    # --- 2. Load the Trained Model ---
    # Initialize the network architecture
    model = simplenetv1_small_m1_05(
            in_chans=AMOUNT_INPUT_FRAMES,
            num_classes=env.amount_actions,
        ).to(device)
    # Load the learned weights from the file
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()

    print("Model loaded successfully. Starting simulation...")
    state = env.reset()

    # --- 3. Setup Video Recording ---
    # Get the dimensions of the first frame to initialize the video writer
    first_frame = raw_env.render()
    height, width, layers = first_frame.shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID'
    video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, FRAME_RATE, (width, height))


    # --- 4. Run the Simulation Loop ---
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Get the rendered frame from the environment for display and recording
        frame_to_render = raw_env.render()
        # The environment renders in RGB, but OpenCV uses BGR. We must convert it.
        bgr_frame = cv2.cvtColor(frame_to_render, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow(f'{ENV_NAME} Simulation', bgr_frame)
        # Write the frame to the video file
        video_writer.write(bgr_frame)

        # Decide on the action
        action = model.act(state) # In evaluation, we use the best action

        # Execute the action in the environment
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        # Add a small delay to make it watchable and control speed
        time.sleep(1 / FRAME_RATE)

        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Simulation stopped by user.")
            break

    # --- 5. Cleanup ---
    print(f"Episode finished with a total reward of: {total_reward}")
    print(f"Video saved as '{VIDEO_FILENAME}'")
    env.env.close()
    cv2.destroyAllWindows()
    video_writer.release()


if __name__ == "__main__":
    # --- Use argparse to get the model path from the command line ---
    parser = argparse.ArgumentParser(description="Test a trained DQN agent for Atari Pong.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model file (.pth)."
    )
    args = parser.parse_args()

    test_agent(args.model_path)