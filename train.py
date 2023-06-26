# python 3.8 reinforcement learning
import numpy as np
#import keras.backend.tensorflow_backend as backend
#from keras.models import Sequential
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
#import tensorflow as tf
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from poker_game import game
from player_class import Player


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 200  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'poker64_32'
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20

# Exploration settings
# new model
epsilon = 1  # not a constant, going to be decayed
# load model
#epsilon = 0.99 ** 200

#EPSILON_DECAY = 0.99975
EPSILON_DECAY = 0.996
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


class PokerEnv:
    OBSERVATION_SPACE_VALUES = (4, )#(1, 4)
    ACTION_SPACE_SIZE = 41

    def __init__(self):
        Player('Alice', 1000, 'AI')
        Player('Bob', 1000, 'deepAI')

    def reset(self):
        player_list = Player.player_list_chair
        for player in player_list:
            player.next_game()
        self.game_generator = game()
        observation, _, _, _ = next(self.game_generator)
        return observation

    def step(self, action):
        observation, reward, done, action_used = self.game_generator.send(action)
        return observation, reward, done, action_used


env = PokerEnv()

# For stats
ep_rewards = [0]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('modelsPoker'):
    os.makedirs('modelsPoker')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        # self._write_logs(stats, self.step)
        # tf.summary.scalar('loss', stats['loss'], step=self.step)
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logsPoker/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def back_to_model(self, old_model):
        self.model.set_weights(old_model.get_weights())
        self.target_model.set_weights(old_model.get_weights())

    def create_model(self):
        model = Sequential()

        model.add(Dense(64, input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Dense(32))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE: # when False procedure below, 10000 dochodzi do 899 i wypierdala bład
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) #/ 10 # czy tutaj by nie trzeba zmienic
        current_qs_list = self.model.predict(current_states, verbose=0)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) #/ 10
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        #self.model.fit(np.array(X) / 10, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
         #              callbacks=[self.tensorboard] if terminal_state else None)

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        #return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0] # state = (1,1,1,1)
        #print(np.array(state) / 10)
        observation = np.array(state).reshape(1, 4) #/ 10
        return self.model.predict(observation, verbose=0)
        #return self.model.predict(np.array(state) / 10)

agent = DQNAgent()
#old_model = load_model('modelsPoker/poker__1000.00max_-161.00avg_-1025.00min__1658267408.model')
#agent.back_to_model(old_model)

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
#for episode in tqdm(range(1, EPISODES + 1)):
    #print(episode)
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        '''
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            #action = np.argmax(agent.get_qs(current_state))
            action = agent.get_qs(current_state)# to jest wektor prawdopodobieństw
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)'''
        action = agent.get_qs(current_state)

        action = np.insert(action, 0, epsilon) # [epsilon, predict vector]
        new_state, reward, done, action_used = env.step(action)
        action = action_used
        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        #if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        #    env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value

        agent.model.save(f'modelsPoker/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)