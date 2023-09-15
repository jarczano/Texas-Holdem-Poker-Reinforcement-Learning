from game.player_class import Player
from game.bot import AI
from game.deepAI import probability_win
from setting import BB, show_game

import numpy as np


def auction(common_cards=None):
    """
    Function made a auction: prepare available options for players, ask players about his decision
    and receive his responses
    :param common_cards: list of common cards, if stage game is preflop then common cards is None
    :return:
    """

    # get live player list
    player_list = Player.player_list
    player_list = [player for player in player_list if player.live]

    number_decisions = sum([player.decision for player in player_list])
    number_player = len(player_list)

    every_fold = False

    # process until every player made decision or every except one fold
    while number_decisions != number_player and not every_fold:

        for player in player_list:

            if not player.decision and player.live:

                # list how much $ players bet in round
                input_stack_list = [player.input_stack for player in player_list]

                # list how much $ players bet in one auction, after each auction attribute bet_auction will be reset
                bet_list = [player.bet_auction for player in player_list]

                # Create a set of options for the player
                dict_options = {'fold': True, 'all-in': True, 'call': False, 'check': False, 'raise': False}

                raise_list = sorted(input_stack_list, reverse=True)
                bet_list = sorted(bet_list, reverse=True)

                # calculate call value, min and max value raise
                call_value = max(input_stack_list) - player.input_stack
                min_raise = call_value + bet_list[0] - bet_list[1]
                if min_raise < BB:
                    min_raise = BB
                max_raise = player.stack

                # activate available option for player
                if player.input_stack == max(input_stack_list):
                    dict_options['check'] = True
                elif player.stack > call_value:
                    dict_options['call'] = True
                if player.stack > min_raise:
                    dict_options['raise'] = True

                pot = sum(raise_list)

                # ask player for decision
                if player.kind == 'human':
                    options = ': '
                    for option in dict_options:
                        if dict_options[option]:
                            options += option + '   '

                    decision = input(player.name + options).split()

                elif player.kind == 'AI':
                    # this function returns choose of AI
                    #  n_fold = [gamer.live and gamer.alin for gamer in player_list].count(True)
                    #  n_player_in_round = number_player - n_fold
                    n_player_in_round = 2

                    bot = AI(player.cards, dict_options, call_value, min_raise, max_raise, pot, n_player_in_round,
                             common_cards)
                    decision = bot.decision()
                    #  print(player.name, decision)

                elif player.kind == 'deepAI':

                    p_win, p_tie = probability_win(player.cards, 2, common_cards)

                    if common_cards is None:
                        stage_round = [1, 0, 0, 0]
                    elif len(common_cards) == 3:
                        stage_round = [0, 1, 0, 0]
                    elif len(common_cards) == 4:
                        stage_round = [0, 0, 1, 0]
                    else:
                        stage_round = [0, 0, 0, 1]

                    observation = [p_win, p_tie, pot / 2000]
                    observation.extend(stage_round)

                    action = yield observation, player.reward, False, player.action_used
                    player.reward = 0

                    epsilon = action[0]
                    DNN_answer = action[1:]

                    # new try

                    # select option with highest propability
                    if np.random.random() > epsilon:
                        best_action_index = np.argmax(DNN_answer)
                    else:
                        best_action_index = np.random.randint(0, 40)

                    # round best action to available best action
                    # there are 5 possible set of action
                    optimal_bet = best_action_index * 25
                    if optimal_bet > player.stack:
                        optimal_bet = player.stack

                    if dict_options['check'] and dict_options['raise']:
                        #  print('set 1')
                        if optimal_bet < abs(optimal_bet - min_raise):
                            decision = ['check']
                        elif min_raise <= optimal_bet <= max_raise:
                            decision = ['raise', optimal_bet]
                        else:
                            decision = ['raise', min_raise]

                    # 2 set
                    elif dict_options['call'] and dict_options['raise']:
                        #  print('set 2')
                        if optimal_bet < abs(optimal_bet - call_value):
                            decision = ['fold']
                        elif abs(optimal_bet - call_value) < abs(optimal_bet - min_raise):
                            decision = ['call']
                        elif min_raise <= optimal_bet <= max_raise:
                            decision = ['raise', optimal_bet]
                        else:
                            decision = ['raise', min_raise]

                    # 3 set
                    elif dict_options['call'] and not dict_options['raise']:
                        #  print('set 3')
                        if optimal_bet < abs(call_value - optimal_bet):
                            decision = ['fold']
                        elif abs(optimal_bet - player.stack) < abs(optimal_bet - call_value):
                            decision = ['all-in']
                        else:
                            decision = ['call']

                    # 4 set
                    elif dict_options['check'] and not dict_options['raise']:
                        #  print('set 4')
                        if optimal_bet < abs(optimal_bet - player.stack):
                            decision = ['check']
                        else:
                            decision = ['all-in']

                    # 5 set
                    elif not dict_options['call'] and not dict_options['check'] and not dict_options['raise']:
                        #  print('set 5')
                        if optimal_bet < abs(optimal_bet - player.stack):
                            decision = ['fold']
                        else:
                            decision = ['all-in']

                    # processing action

                    # convert decision to action_used
                    if decision == ['fold'] or decision == ['check']:
                        action_used = 0
                    elif decision == ['call']:
                        action_used = (call_value / 25) + 1
                    elif decision == ['all-in']:
                        action_used = (player.stack / 25) + 1
                    elif decision[0] == 'raise':
                        action_used = (decision[1] / 25) + 1

                    player.action_used = int(action_used)

                # Processing of player decision

                if decision[0] == 'raise':
                    chips = int(decision[1])
                decision = decision[0]

                if show_game:
                    if decision == 'raise':
                        print("{} stack: {} decision: {} {}".format(player.name, player.stack,decision, chips))
                    else:
                        print("{} decision: {}".format(player.name, decision))

                if decision == 'call':
                    chips = max(input_stack_list) - player.input_stack
                    if player.stack > chips:
                        player.drop(chips)
                    else:
                        player.drop(player.stack)
                        player.allin()

                elif decision == 'fold':
                    player.fold()

                elif decision == 'check':
                    player.decision = True

                elif decision == 'all-in':
                    player.drop(player.stack)
                    for gamer in player_list:
                        # if any of player bets all-in, then each player who not bet all-in and has bet less than
                        # that player all-in, will have to make the decision again
                        if gamer.live and gamer.decision and gamer.input_stack < player.input_stack:
                            gamer.decision = False
                    player.allin()

                elif decision == 'raise':
                    for gamer in player_list:
                        # a czy tutaj nie musi byc tak jak w przypadku all in ze gracze ktorzy mniej postawili
                        if gamer.live and gamer.decision:
                            gamer.decision = False
                    if player.stack > chips:
                        player.drop(chips)
                    else:
                        player.drop(player.stack)
                        player.allin()

            # check if the every except one player fold then don't ask him about decision
            sum_live = 0
            sum_alin = 0
            for gamer in player_list:
                sum_live += gamer.live
                sum_alin += gamer.alin
            if sum_live == 1 and sum_alin == 0:
                every_fold = True
                break
        number_decisions = sum([player.decision for player in player_list])

    # After auction players who fold or all-in have no decision made until the next round
    for player in player_list:
        player.next_auction()
        if player.live:
            player.decision = False

