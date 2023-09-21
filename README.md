# Texas-Holdem-Poker-Reinforcement-Learning

## Overview:
This project aims to teach a deep neural network model to play Texas Hold'em Poker in the 2-player version. Using reinforcement learning techniques, we train an artificial intelligence agent to make strategic decisions in a simulated poker game. The project is implemented in Python and leverages TensorFlow and Keras for the development and training of the AI model.

## Project Description

This project is a comprehensive exploration of training a Deep Neural Network (DNN) model to play Texas Hold'em Poker effectively in a two-player environment. It consists of two main components: a poker game simulator and the neural network model training.

### Poker Game Simulator

The first component of the project is a poker game simulator, which serves as an environment generator. This simulator provides a detailed representation of the poker game state and allows users to interact with the game by sending actions to it. Key features include:

- Detailed game state information (player hands, community cards, pot size, etc.).
- User interactions: Send actions like folding, betting, or raising.
- Realistic and dynamic game environment.

The poker game simulator forms the foundation for training the AI model, offering a realistic environment for learning and strategy development.

### Neural Network Model Training

The second component involves building and training a Deep Neural Network (DNN) model to play Texas Hold'em Poker intelligently. Two primary training scripts are included:

- **train RL.py:** Designed for single-threaded training, allowing the model to learn strategies through poker games against opponents, such as bots or other versions of itself.

- **train RL multiprocess.py:** Utilizes multiprocessing to accelerate training by running multiple poker games simultaneously.

During the training process, the DNN model learns to make informed decisions based on the game state, opponent behavior, and historical data. The goal is to develop effective poker-playing strategies that maximize winnings and minimize losses.

This project serves as an exciting experiment in the field of reinforcement learning and AI-driven gaming. Teaching a DNN model to play poker offers insights into AI's capabilities and limitations in strategic decision-making scenarios. The model's performance can be fine-tuned and evaluated over multiple training sessions, contributing to the development of more advanced AI gaming systems.

Whether you're a poker enthusiast, an AI researcher, or simply curious about the convergence of AI and gaming, this project provides a valuable opportunity to explore reinforcement learning and witness a neural network's journey to outsmart opponents in a classic card game.

## Usage:

- Clone the repository: `git clone https://github.com/jarczano/Sudoku-Solver-Web-App`
- Install the requirements: `pip install -r requirements.txt`
- Run `train RL.py`: `python train RL.py`
- or
- Run `train RL multiprocess.py`: `python train RL multiprocess.py`

The trained model will be saved as an .h5 file in the /models folder

## Technologies:
- Numpy
- Keras
- Tensorflow

## Related Projects:
- Texas Holdem Poker Web Game https://github.com/jarczano/Texas-Holdem-Poker-Web-App
## License:
- MIT

## Author:
- Jarosław Turczyn
