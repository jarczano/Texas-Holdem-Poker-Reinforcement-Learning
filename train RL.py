from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, LSTM, Attention, Input, Layer
import numpy as np
from player_class import Player
from poker_game import game
from keras.optimizers import Adam
import time
import tensorflow


physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


def create_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(7,)),
        Dense(32, activation='relu'),
        Dense(41)])
    model.compile(optimizer=Adam(), loss='mse')
    return model


def train(model):
    """
    The function teaches the model how to play heads up poker
    :param model: model to train
    :return: saves the learned model as file .h5
    """
    model_target = create_model()
    model_target.set_weights(model.get_weights())

    path_save_model = r"models"

    # Learning parameters
    epsilon = 0.9
    epoch_start = 0  # change if retrain
    n_epoch = 10_000
    epsilon_decay = 0.999
    min_epsilon = 0.1
    discount = 0.9

    Player('Alice', 1000, 'AI')
    Player('Bob', 1000, 'deepAI')
    n_win_games = 0
    game_result_list = []
    game_length_list = []

    n_epoch_update_model = 20
    n_epoch_save_model = 100

    for epoch in range(epoch_start, n_epoch):

        for player in Player.player_list:
            player.stack = 1000

        game_instance = game()

        # Update target model
        if epoch % n_epoch_update_model == 0:
            model_target.set_weights(model.get_weights())

        states = []
        actions = []
        rewards = []
        next_states = []
        targets = []

        state, _, _, _ = next(game_instance)

        done = False
        while not done:

            # instead of selecting here which action is to be taken, we must send the entire action vector and epsilon,
            # because here it is not known which actions are possible to take at a given moment
            action_vector = model.predict(np.array(state).reshape(1, 7))

            targets.append(action_vector)
            action_vector = np.insert(action_vector, 0, epsilon)

            # send action vector to game
            # action_used denotes which action was really played in the previous turn
            new_state, reward, done, action_used = game_instance.send(action_vector)

            states.append(state)
            actions.append(action_used)
            rewards.append(reward)
            next_states.append(new_state)

            state = new_state

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        # Conversion lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        targets = np.squeeze(np.array(targets), axis=1)
        next_states = np.array(next_states)

        # Calculate target Q-values
        next_q_values = model_target.predict(next_states[:-1])
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards[:-1] + discount * max_next_q_values

        target_q_values = np.append(target_q_values, rewards[-1])

        for i in range(len(targets)):
            targets[i][actions[i]] = target_q_values[i]

        total_reward = sum(rewards)

        if total_reward > 0:
            game_result_list.append(1)
            n_win_games += 1
        else:
            game_result_list.append(0)

        win_ratio = n_win_games / (epoch + 1) * 100

        game_length_list.append(len(rewards))

        # Train model on batch
        model.train_on_batch(states, targets)

        print("epoch {}, rewards: {}, win games {} %".format(epoch, np.sum(rewards), win_ratio))
        # Save model
        if epoch % n_epoch_save_model == 0 and epoch != 0:

            model_name = "model_{}epoch-{}".format(epoch, time.time())
            model.save(str(path_save_model) + r"/{}.h5".format(model_name))
            # Create information file about model

            # param to retrain and model info
            with open(path_save_model + r"/param_{}.txt".format(model_name), 'w') as file:
                file.write("Architecture model: \n")
                file.write("Number epoch: {}\n".format(epoch))
                file.write("Epsilon: {}\n".format(epsilon))
                model.summary(print_fn=lambda x: file.write(x + '\n'))

            # stats
            with open(path_save_model + r"/stats_{}.txt".format(model_name), 'w') as file:
                file.write("Win/lose: {}\n".format(game_result_list))
                file.write("Game length: \n{}".format(game_length_list))


if __name__ == '__main__':
    model = create_model()
    # or load model
    # model = load_model('models/xxx.h5')
    train(model)
