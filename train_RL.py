from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, LSTM, Attention, Input, Layer
import numpy as np
from player_class import Player
from poker_game import game
from keras.optimizers import Adam
import time
import tensorflow
path_save_model = r"models"

physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) >= 1:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

'''
model = Sequential([
    #LSTM(64, input_shape=(4,1)),
    Dense(128, activation='relu', input_shape=(9,)),
    Attention(),
    Dense(64, activation='relu'),
    Dense(41)])
'''

'''
inputs = Input(shape=(9,))
x = Dense(128, activation='relu')(inputs)
attention = Attention()([x, x])  # Użycie warstwy uwagi z dwoma argumentami wejściowymi
x = attention[0]  # Wyjście z warstwy uwagi
x = Dense(64, activation='relu')(x)
outputs = Dense(41)(x)
model = Model(inputs=inputs, outputs=outputs)
'''
"""
model = Sequential([
    Dense(128, activation='relu', input_shape=(9,)),
    Attention(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(41)
])
"""
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention = Attention()
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        query = inputs
        value = inputs
        return self.attention([query, value])
'''
model = Sequential([
    Dense(128, activation='relu', input_shape=(9,)),
    AttentionLayer(),
    Dense(64, activation='relu'),
    Dense(41)
])
model.compile(optimizer=Adam(), loss='mse')
'''
model = load_model('models/model_11099epoch-1688823889.5684729.h5')
epoch_start = 11099
n_epoch = 15_000
epsilon = 0.1
epsilon_decay = 0.999
min_epsilon = 0.1
discount = 0.9

Player('Alice', 1000, 'AI')
Player('Bob', 1000, 'deepAI')
epochs_info = []
n_win_games = 0
game_result_list = []
game_length_list =[]
list_win_ratio = []


for epoch in range(epoch_start, n_epoch):

    for player in Player.player_list:
        player.stack = 1000
    game_instance = game()

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

    for i in range(len(targets)):
        targets[i][actions[i]] = target_q_values[i]

    total_reward = sum(rewards)

    if total_reward > 0:
        game_result_list.append(1)
        n_win_games += 1
    else:
        game_result_list.append(0)


    win_ratio = n_win_games / (epoch + 1) * 100

    #list_win_ratio.append(win_ratio)
    game_length_list.append(len(rewards))
    #epochs_info.append([total_reward, len(rewards)])

    # Trenowanie modelu na batchu danych
    model.train_on_batch(states, targets)
    #model.fit(states,targets)

    print("epoch {}, rewards: {}, win games {} %".format(epoch, np.sum(rewards), win_ratio))
    if (epoch + 1) % 100 == 0:

        model_name = "model_{}epoch-{}".format(epoch, time.time())
        model.save(str(path_save_model) + r"/{}.h5".format(model_name))
        # Create information file about model

        # param to retrain and model info
        with open(path_save_model + r"/param_{}.txt".format(model_name), 'w') as file:
            file.write("Architecture model: \n")
            file.write("Number epoch: {}\n".format(epoch))
            file.write("Epsilon: {}\n".format(epsilon))
            model.summary(print_fn=lambda x: file.write(x + '\n'))
            #file.write("Win ratio: {}\n".format(list_win_ratio))
            #file.write("Epoch info: \n{}".format(epochs_info))

        # stats
        with open(path_save_model + r"/stats_{}.txt".format(model_name), 'w') as file:
            file.write("Win/lose: {}\n".format(game_result_list))
            file.write("Game length: \n{}".format(game_length_list))

