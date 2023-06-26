import numpy as np
import operator
from player_class import Player


def split_pot():

    # The split_pot function takes one parameter: list of object players.
    # Function changing attribute stack this object and returns how much they win.
    import operator
    player_list = Player.player_list_chair.copy()

    # Remove players which fold
    for player in player_list:
        if player.live is False and player.alin is False:
            # nagroda jezeli foldowach
            if player.kind == "deepAI":
                print("split pot action used1", player.action_used)

                yield np.zeros(4) - 1, -player.input_stack, False, player.action_used
            player_list.remove(player)

    # To calculate reword function needs sorted players list with a descending order and then ascending input stack
    player_list.sort(key=operator.attrgetter('input_stack'))
    player_list.sort(key=operator.attrgetter('score'), reverse=True)
    n = len(player_list)
    player_score, input_stack = [], []

    for player in player_list:
        player_score.append(player.score)
        input_stack.append(player.input_stack)

    win_list = [0] * n
    input_in_game = [0] * n

    # Calculates how many players will be given back the chips they have put into the main pot
    for i in range(n):
        if player_score[i] == max(player_score):
            input_in_game[i] = input_stack[i]
        else:
            aux = [0] * n
            new_input = [0] * n
            for j in range(n):
                if player_score[j] != player_score[i]:
                    aux[j] = 1
            for j in range(n):
                new_input[j] = aux[j] * input_stack[j]
            if input_stack[i] - max(new_input[0:i]) < 0:
                input_in_game[i] = 0
            else:
                input_in_game[i] = input_stack[i] - max(new_input[0:i])

    # Calculates how many each player wins the chips
    for i in range(n):
        number_division = player_score[i:].count(player_score[i])
        for j in range(i + 1, n):
            if player_score[i] > player_score[j]:
                if input_stack[i] >= input_stack[j]:
                    win_list[i] += input_stack[j] / number_division
                    input_stack[j] = 0
                elif input_stack[i] < input_stack[j]:
                    win_list[i] += input_stack[i] / number_division
                    input_stack[j] -= input_stack[i]
        if number_division > 1:
            for k in range(i + 1, n):
                if player_score[i] == player_score[k]:
                    win_list[k] = win_list[i]
                    input_stack[k] -= input_stack[i]

    # Sum of chips returned and won
    list_winner = []
    for i in range(n):
        win_value = input_in_game[i] + win_list[i]
        list_winner.append((player_list[i], int(win_list[i])))
        # tutaj mozna by if plaer.kind == "deepAI" yield ...
        # potrzebuje jeszcze ile wjebał
        # ale tutaj pomija tych co fold - to chuj dopiszemy
        player_list[i].win(win_value)
        if player_list[i].kind == 'deepAI':
            print("split pot action used2", player_list[i].action_used, player_list[i].name)
            yield np.zeros(4) - 1, win_value - player_list[i].input_stack, False, player_list[i].action_used

        #
        #print(player_list[i].name, 'won ',win_value - player_list[i].input_stack)
        #print(player_list[i].name, player_list[i].stack, player_list[i].input_stack)
    return list_winner
    #print('wklad wlasny: ', wkladwlasny)
    #print('win: ', win_list)
    #print('input: ', input_stack)


def one_player_win():

    #  Function changing player stack who win, and return list tuple who win and how much

    player_list = Player.player_list_chair.copy()
    list_winner = []
    for player in player_list:
        if player.live or player.alin:
            win_value = sum([player.input_stack for player in player_list])
            player.win(win_value)
            list_winner.append((player, win_value - player.input_stack))
            if player.kind == 'deepAI':
                print("split pot action used3", player.action_used)
                yield np.zeros(4) - 1, win_value - player.input_stack, False, player.action_used
        else:
            # Reward for fold for deepAI
            if player.kind == "deepAI":
                print("split pot action used4", player.action_used)
                yield np.zeros(4) - 1, -player.input_stack, False, player.action_used
    return list_winner


def change_players_positions(shift):
    # Function change each player position
    # order in Player.player_list are changed


    player_list = Player.player_list
    number_players = len(player_list)
    for player in player_list:
        player.position = (player.position + shift) % number_players
    player_list.sort(key=operator.attrgetter('position'))


'''
player_score = [160, 140, 140, 140, 130]
input_stack = [200, 100, 1000, 1200, 800]


#split_pot(player_score, input_stack)





import PlayerClass

start_stack = 5000
Jarek = PlayerClass.Player('Jarek', start_stack, 0)
Kuba = PlayerClass.Player('Kuba', start_stack, 1)
Darek = PlayerClass.Player('Darek', start_stack, 2)
Marek = PlayerClass.Player('Marek', start_stack, 3)
Player_list = [Jarek, Kuba, Darek, Marek]
stack_list_before = [0] * 4
for i in range(4):
    stack_list_before[i] = Player_list[i].stack
print('before: ', stack_list_before)
Jarek.stack -= 250
stack_list_after = [0] * 4
for i in range(4):
    stack_list_after[i] = Player_list[i].stack

print('after: ', stack_list_after)

import PlayerClass
import operator
start_stack = 5000
Jarek = PlayerClass.Player('Jarek', start_stack, 0)
Kuba = PlayerClass.Player('Kuba', start_stack, 1)
Darek = PlayerClass.Player('Darek', start_stack, 2)
Marek = PlayerClass.Player('Marek', start_stack, 3)
Player_list = [Jarek, Kuba, Darek, Marek]

small_blind = 25
big_blind = 50
Marek.drop(big_blind)
Darek.drop(small_blind)
first_auction(Player_list)
first_auction(Player_list)

Jarek.score = 100
Jarek.drop(150)
Kuba.score = 100
Kuba.drop(100)
Marek.score = 160
Marek.drop(500)
Darek.score = 90
Darek.drop(600)
split_pot(Player_list)
'''
#for player in Player_list:
#    print(player.name, player.stack)