from player_class import Player
from poker_round import poker_round
from split_pot import change_players_positions
import numpy as np

# Start settings

# Creating players


def game():
    #Player('Alice', 1000, 'AI')
    #Player('Bob', 1000, 'deepAI')

    player_list_chair = Player.player_list_chair

    # Play a game until there is only one player left
    #while len(player_list_chair) > 1:
    end = False
    while not end:

        # Play a round
        yield from poker_round()

        # Shift the button to the next player
        change_players_positions(shift=1)

        # Reset properties for each player
        [player.next_round() for player in player_list_chair]

        # Remove players who have lost
        #[player_list_chair.remove(player) for player in player_list_chair if player.stack == 0]

        # Check if players has money
        #[True for player in player_list_chair in player.stack == 0]
        for player in player_list_chair:
            if player.stack == 0:
                end = True

    #observation = [p_win, p_tie, pot, stage, players_without_decision, action_info[0],
    #              action_info[1]]  # len vector 7, one hot encoding ?
    # new_state, reward, done = env.step(action)
    # tutaj tez by musialo zwrocic jaki był ostania action
    for player in player_list_chair:
        if player.kind == 'deepAI':
            print("poker_game action used", player.action_used)
            yield np.zeros(4) - 1, 0, True, player.action_used


'''
# to juz jest moja funckja reset()
game_generator = game()
observation, reward, done = next(game_generator)
print(f'observation {observation}, reward {reward}, done {done}')


game_generator = game()
observation, reward, done = next(game_generator)
print(f'observation {observation}, reward {reward}, done {done}')
'''
'''
# step
#for i in range(10):
done = False
while not done:
    action = np.random.rand(43)
    observation, reward, done = game_generator.send(action)
    print(f'observation {observation}, reward {reward}, done {done}')
    # teraz jeszcze trzeba by ja uzupełnic o nagrode i done


'''
#observation = game_generator.send(3)
#observation = next(game_generator)
#print(observation)



