from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, LSTM, Attention, Input, Layer
import numpy as np
from player_class import Player
from poker_game import game
from keras.optimizers import Adam
import tensorflow
path_save_model = r"models"
import time


from multiprocessing import Process, Queue


def create_model():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(7,)),
        Dense(32, activation='relu'),
        Dense(41)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model


def one_game(epsilon, weight_model, queue):
    discount = 0.9
    model = create_model()
    model.set_weights(weight_model)

    #print("weight from game", model.get_weights()[0][0])
    #print(epsilon)
    #print(weight_model)
    Player('Alice', 1000, 'AI')
    Player('Bob', 1000, 'deepAI')
    #for player in Player.player_list:
    #    player.stack = 1000

    #print("len player list ", len(Player.player_list))

    game_instance = game()
    result = []
    states = []
    actions = []
    rewards = []
    next_states = []

    state, _, _, _ = next(game_instance)

    done = False

    while not done:
        # inaczej zamist tutaj losowo wybierac to trzeba przeslac wektor action i espilona do gry bo tam zostaje wybrana akcja

        action_vector = model.predict(np.array(state).reshape(1, 7))
        action_vector = np.insert(action_vector, 0, epsilon)

        # send action vector to game
        new_state, reward, done, action_used = game_instance.send(action_vector)

        states.append(state)
        actions.append(action_used)
        rewards.append(reward)
        next_states.append(new_state)

        state = new_state

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

    for i in range(len(targets)):
        targets[i][actions[i]] = target_q_values[i]

    total_reward = sum(rewards)

    result.append(states)
    result.append(targets)
    #model.train_on_batch(states, targets)

    if total_reward > 0:
        result.append(1)
    else:
        result.append(0)

    queue.put(result)

physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':


    # Start settings

    epsilon = 0.1
    epsilon_decay = 0.9996
    min_epsilon = 0.1

    epochs_info = []
    n_win_games = 0
    game_result_list = []
    game_length_list = []
    list_win_ratio = []
    history = []

    #model1 = create_model()
    model1 = load_model('models/model_15003epoch-1688848258.370219.h5')
    #model2 = Sequential.from_config(model1.get_config())
    #model2.set_weights(model1.get_weights())
    model2 = create_model()
    model2.set_weights(model1.get_weights())

    weight_model1 = model1.get_weights()

    n_process = 5
    processes = []
    queue_list = []
    for i in range(n_process):
        queue_list.append(Queue())
        #print("create {}".format(i))
        #processes.append(Process(target=one_game, args=(epsilon, queue_list[i],)))
        processes.append(Process(target=one_game, args=(epsilon, weight_model1, queue_list[i],)))

    for j in range(n_process):
        #print("start {}".format(j))
        processes[j].start()

    total_iterations = 20_000
    completed_iterations = 15003
    X = []
    y = []

    run_train = False
    update_model = False
    save_file = False

    while completed_iterations < total_iterations:

        for process in range(n_process):
            if not processes[process].is_alive():
                processes[process] = Process(target=one_game, args=(epsilon, weight_model1, queue_list[process],))
                #processes[process] = Process(target=one_game, args=(epsilon, queue_list[process],))
                #print("process {} start".format(process))
                processes[process].start()
                completed_iterations += 1
                print("completed: ", completed_iterations)
                if completed_iterations % 5 == 0:
                    run_train = True
                if completed_iterations % 20 == 0:
                    update_model = True
                if completed_iterations % 100 == 0:
                    save_file = True

                if epsilon > min_epsilon:
                    epsilon *= epsilon_decay
        for q in range(n_process):
            if not queue_list[q].empty():
                result = queue_list[q].get()
                X_game, y_game, result_game = result
                X.append(X_game)
                y.append(y_game)
                game_result_list.append(result_game)
                #print("process {} win/lose {}".format(q, result_game))

        # tu jest ten problem ze to dziala non stop i warunek bedzie spelniony przez n wykonan dopoki sie nie zmieni na inny

        if run_train and (len(X) > 0): # ten warunek został przesuniety
            print('train')
            #print('run train')
            run_train = False
            # tutaj jest taka luka ze podczas szkolenia bedą X zapisywane do starego X a po szkoleniu zostana wymazane
            # czyli przed szkoleniem trzeba zrobic nowa tablice do ktorej się bedą x zapisywały
            # a moze nie, czy ten kod sie tutaj nie zatrzymuje normalnie ?, ten proces se działa ale zapisane do X zostanie dopiero gdy kod tam wroci
            X = np.vstack(X)
            y = np.vstack(y)
            #model2.fit(X,y, verbose=1)
            model2.train_on_batch(X, y)
            model1.set_weights(model2.get_weights())
            weight_model1 = model1.get_weights()
            #print("new weight after train", model2.get_weights()[0][0])
            X = []  # tylko czy tutaj mogą mi nie wyminąć sie jakies dane
            y = []

        if update_model:
            update_model = False
            model1.set_weights(model2.get_weights())
            weight_model1 = model1.get_weights()

        if save_file:
            save_file = False
            model_name = "model_{}epoch-{}".format(completed_iterations, time.time())
            model2.save(str(path_save_model) + r"/{}.h5".format(model_name))
            # Create information file about model

            # param to retrain and model info
            with open(path_save_model + r"/param_{}.txt".format(model_name), 'w') as file:
                file.write("Architecture model: \n")
                file.write("Number epoch: {}\n".format(completed_iterations))
                file.write("Epsilon: {}\n".format(epsilon))
                model2.summary(print_fn=lambda x: file.write(x + '\n'))
                # file.write("Win ratio: {}\n".format(list_win_ratio))
                # file.write("Epoch info: \n{}".format(epochs_info))

            # stats
            with open(path_save_model + r"/stats_{}.txt".format(model_name), 'w') as file:
                file.write("Win/lose: {}\n".format(game_result_list))
                file.write("Game length: \n{}".format(game_length_list))

    for k in range(len(processes)):
        print("join {}".format(k))
        processes[k].join()




