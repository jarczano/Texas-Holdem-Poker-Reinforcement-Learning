'''
def game():
    counter_state = 0
    while True:
        action = yield "state {}".format(counter_state)
        counter_state += 1 * action


game = game()
state = next(game)
print("star state:",state)

action = 2
state2 = game.send(2)
print("state 2", state2)
'''
import numpy as np
import random
from player_class import Player
from poker_game import game
Player('Alice', 1000, 'AI')
Player('Bob', 1000, 'deepAI')
game = game()
state = next(game)
print('start state: ', state)
action_vector = np.random.rand(41)
print("gen action vector")
state2 = game.send(action_vector)
print("state 2: ", state2)


ac = np.random.choice([i for i in range(41)])
print(ac)
#action  = np.random.choice()