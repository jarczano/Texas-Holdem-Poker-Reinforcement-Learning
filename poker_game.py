from player_class import Player
from poker_round import poker_round
from split_pot import change_players_positions
import numpy as np


def game():
    """
    This function works as generator.
    observation, player.reward, False, player.action_used
    :return: yield 'wait_for_player_decision' when is turn for client. yield 'end' at the end of the game
    """
    player_list_chair = Player.player_list_chair

    end = False
    while not end:

        # Play a round
        yield from poker_round()

        # Shift the button to the next player
        change_players_positions(shift=1)

        # Reset properties for each player
        [player.next_round() for player in player_list_chair]

        # Check if players has money
        for player in player_list_chair:
            if player.stack == 0:
                end = True

    for player in player_list_chair:
        if player.kind == 'deepAI':
            #  print("poker_game action used", player.action_used)
            yield np.zeros(7) - 1, player.reward, True, player.action_used





