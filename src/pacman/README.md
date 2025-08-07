# Pseudocode

## Q_NeuralNet
    - The heart of the RL agent, mapping inputs to outputs (estimated Q function)
    1) Original Architecture
    2) More sophisticated Architecture with recent advancements in CV

## DQN_AGENT
    - Main Class that is getting called from the main
    - Wraps Q_NeuralNet and has a train(), act() and memorize() function
    - Implements boltzman equation

## REPLAY_MEMORY
    - Saves all the previous states in memory
    - Should have a sample() method

## Preprocessor
    - Crops the image, transfers to grayscale etc.