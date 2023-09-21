from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from keras.optimizers import Adam
import tensorflow
import time
from multiprocessing import Process, Queue

from game.player_class import Player
from game.poker_game import game


physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


def create_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(7,)),
        Dense(32, activation='relu'),
        Dense(41)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model


def one_game(epsilon, weight_model, weight_model_target, queue):
    """
    :param epsilon: epsilon-greedy
    :param weight_model: weights main model
    :param weight_model_target: weights target model
    :param queue: Queue object from multiprocessing
    :return:
    """

    discount = 0.9
    model = create_model()
    model.set_weights(weight_model)

    model_target = create_model()
    model_target.set_weights(weight_model_target)

    Player('Alice', 1000, 'AI')
    Player('Bob', 1000, 'deepAI')

    game_instance = game()
    result = []
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

    # Conversion lists to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    targets = np.squeeze(np.array(targets), axis=1)
    next_states = np.array(next_states)

    # Calculate target Q-values
    if next_states.shape[0] > 1:
        next_q_values = model_target.predict(next_states[:-1])
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards[:-1] + discount * max_next_q_values

        target_q_values = np.append(target_q_values, rewards[-1])
    else:

        target_q_values = rewards

    for i in range(len(targets)):
        targets[i][actions[i]] = target_q_values[i]

    total_reward = sum(rewards)

    result.append(states)
    result.append(targets)

    if total_reward > 0:
        result.append(1)
    else:
        result.append(0)

    queue.put(result)


if __name__ == '__main__':

    path_save_model = r"models"

    # Learning parameters
    epsilon = 0.1
    epsilon_decay = 0.9996
    min_epsilon = 0.1

    epochs_info = []
    n_win_games = 0
    game_result_list = []
    game_length_list = []
    list_win_ratio = []
    history = []

    model = create_model()
    # or load model
    #  model = load_model('models/xxx.h5')

    model_target = create_model()
    model_target.set_weights(model.get_weights())

    weight_model = model.get_weights()
    weight_model_target = model_target.get_weights()

    n_process = 5
    processes = []
    queue_list = []

    for i in range(n_process):
        queue_list.append(Queue())
        processes.append(Process(target=one_game, args=(epsilon, weight_model, weight_model_target, queue_list[i],)))

    for j in range(n_process):
        processes[j].start()

    total_iterations = 20_000
    completed_iterations = 0  # change if training continues
    X = []
    y = []

    run_train = False
    update_model = False
    save_file = False

    n_epoch_train_model = 5
    n_epoch_update_model = 20
    n_epoch_save_model = 100

    while completed_iterations < total_iterations:

        # Start processes that have ended
        for process in range(n_process):
            if not processes[process].is_alive():

                # Get data from process which end
                result = queue_list[process].get()
                X_game, y_game, result_game = result
                X.append(X_game)
                y.append(y_game)
                game_result_list.append(result_game)

                # Restart the process
                processes[process] = Process(target=one_game, args=(epsilon, weight_model, weight_model_target, queue_list[process],))
                processes[process].start()

                completed_iterations += 1

                print("completed: ", completed_iterations)

                if completed_iterations % n_epoch_train_model == 0 and completed_iterations != 0:
                    run_train = True
                if completed_iterations % n_epoch_update_model == 0 and completed_iterations != 0:
                    update_model = True
                if completed_iterations % n_epoch_save_model == 0 and completed_iterations != 0:
                    save_file = True

                if epsilon > min_epsilon:
                    epsilon *= epsilon_decay

        if run_train:

            run_train = False
            X = np.vstack(X)
            y = np.vstack(y)

            model.train_on_batch(X, y)
            weight_model = model.get_weights()

            X, y = [], []

        if update_model:
            update_model = False
            model_target.set_weights(model.get_weights())
            weight_model_target = model_target.get_weights()

        if save_file:
            save_file = False
            model_name = "model_{}epoch-{}".format(completed_iterations, time.time())
            model.save(str(path_save_model) + r"/{}.h5".format(model_name))
            # Create information file about model

            # param to retrain and model info
            with open(path_save_model + r"/param_{}.txt".format(model_name), 'w') as file:
                file.write("Architecture model: \n")
                file.write("Number epoch: {}\n".format(completed_iterations))
                file.write("Epsilon: {}\n".format(epsilon))
                model.summary(print_fn=lambda x: file.write(x + '\n'))
                # file.write("Win ratio: {}\n".format(list_win_ratio))
                # file.write("Epoch info: \n{}".format(epochs_info))

            # stats
            with open(path_save_model + r"/stats_{}.txt".format(model_name), 'w') as file:
                file.write("Win/lose: {}\n".format(game_result_list))
                file.write("Game length: \n{}".format(game_length_list))

    # end processes
    for k in range(len(processes)):
        print("join {}".format(k))
        processes[k].join()




