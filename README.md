# Deep-Learning-Model-for-Atari-Game
This project demonstrates the application of deep learning to enable a computer to play an Atari game. We utilize two distinct machine learning approaches - Deep Q-Networks (DQN) and Double Deep Q-Networks (DDQN) - to maximize the game score.

The DQN model takes multiple frame samples as input and learns from the corresponding output during training. The DDQN model enhances this by using the DQN model alongside another model that evaluates the action at the next state. This dual-model system provides a more robust evaluation of actions, helping to minimize non-optimal predictions that might seem feasible at the current time. In essence, this approach helps avoid overestimation within the model and promotes more consistent results during the deep learning process.

The repository contains several files, each with a distinct role in the model. Here's an overview:

**agent.py**: This file contains the code for the DQN model. It includes all the necessary libraries and initializes the DQN agent, defining hyperparameters, and the memory. The agent utilizes an epsilon-greedy policy for action selection and learns from its actions by training with samples from its memory.

**agent_double.py**: This file extends the DQN model to the DDQN model. It introduces a target network in addition to the policy network used in the DQN model. The target network periodically updates its parameters to match the policy network.

**breakout_ddqn.mp4**: This is a video file showing the Atari game being played by the computer using the DDQN model.

**config.py**: This file defines hyperparameters for DQN agent, memory and training.

**memory.py**: This script contains a ReplayMemory class that stores state, action, reward, and next state for the agent. This memory is used to sample mini-batches of experiences for training.

**model.py**: This file contains the design of the DQN model. It uses PyTorch to construct the model with convolutional layers and a fully connected layer.

**utils.py**: This script contains utility functions such as pre-processing game frames, checking game status, and initializing state history.

Please read the code documentation in each file for more detailed information about the methods and classes.
