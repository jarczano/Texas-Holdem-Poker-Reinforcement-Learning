from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.models import load_model
import numpy as np
from player_class import Player
from poker_game import game


model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(41)])


n_epoch = 100
epsilon = 1
epsilon_decay = 0.99
min_epsilon = 0.1
discount = 0.9
average_rewards = []


for epoch in range(n_epoch):

    Player('Alice', 1000, 'AI')
    Player('Bob', 1000, 'deepAI')
    game = game()

    states = []
    actions = []
    rewards = []
    next_states = []

    state, _, _, _ = next(game)
    total_rewards = 0
    done = False

    while not done:
        # send action to game

        # inaczej zamist tutaj losowo wybierac to trzeba przeslac wektor action i espilona do gry bo tam zostaje wybrana akcja
        action_vector = model.predict(np.array(state).reshape(1,4))
        action_vector = np.insert(action_vector, 0, epsilon)

        # send action vector to game
        new_state, reward, done, action_used = game.send(action_vector)

        actions.append(action_used)
        if done:
            break
        states.append(state)

        rewards.append(reward)
        next_states.append(new_state)

        state = new_state

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    # Konwersja list do numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)

    # Obliczenie docelowych Q-wartości
    next_q_values = model.predict(next_states)
    max_next_q_values = np.max(next_q_values, axis=1)
    target_q_values = rewards + discount * max_next_q_values  # Dyskontowanie przyszłych nagród

    # Przygotowanie danych treningowych
    targets = model.predict(states)
    a = np.arange(len(targets))
    b = actions
    targets[a, b] = target_q_values

    # Trenowanie modelu na batchu danych
    model.train_on_batch(states, targets)

    print("epoch {}, rewards: {}".format(epoch, np.sum(rewards)))



