import random
from player_class import Player
from bot import AI
from deepAI import probability_win
import numpy as np
from setting import BB


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
    last_action = None

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
                    #dict_options[f'raise: {min_raise}-{max_raise}'] = True
                    dict_options['raise'] = True

                pot = sum(raise_list)  # wszystko co jest do wygrania na stole polozone
                pot_table = sum(input_stack_list) - sum(bet_list)  # to co jest w puli zbudowane przed biezaca licytacja

                # ask player for decision
                if player.kind == 'human':
                    options = ': '
                    for option in dict_options:
                        if dict_options[option]:
                            options += option + '   '

                    decision = input(player.name + options).split()

                elif player.kind == 'AI':
                    # this function return choose of AI
                    n_fold = [gamer.live and gamer.alin for gamer in player_list].count(True)
                    n_player_in_round = number_player - n_fold

                    bot = AI(player.cards, dict_options, call_value, min_raise, max_raise, pot, n_player_in_round,
                             common_cards)
                    decision = bot.decision()
                    #print(player.name, decision)

                elif player.kind == 'deepAI':

                    n_fold = [gamer.live and gamer.alin for gamer in player_list].count(True)
                    n_player_in_round = number_player - n_fold

                    p_win, p_tie = probability_win(player.cards, n_player_in_round, common_cards)

                    if common_cards is None:
                        stage = 0
                    elif len(common_cards) == 3:
                        stage = 1
                    elif len(common_cards) == 4:
                        stage = 2
                    else:
                        stage = 3

                    # count how many players will still make a decision
                    players_without_decision = [gamer.decision for gamer in player_list].count(False) - 1

                    # info about opponent action
                    if last_action is None:
                        action_info = [0, 0]
                    elif last_action[0] == 'call':
                        action_info = [1, 0]
                    elif last_action[0] == 'check':
                        action_info = [2, 0]
                    elif last_action[0] == 'raise':
                        action_info = [3, last_action[1]]

                    # calculate min price of bet
                    if dict_options['check']:
                        price_bet = 0
                    elif dict_options['call']:
                        price_bet = call_value
                    else:
                        price_bet = player.stack

                    #observation = [p_win, p_tie, pot/2000, stage/3, players_without_decision, action_info[0]/3, action_info[1]/1000] # len vector 7, one hot encoding ?
                    observation = [p_win, p_tie, pot/2000, price_bet/1000]

                    reward = 0
                    # reset() zwraca to za yield
                    # step() ustawia action na wektor i odsyła odrazu tam gdzie spotka nastepnego yielda
                    #print("przed yield")
                    action = yield observation, reward, False, player.action_used
                    #print("za yield")
                    epsilon = action[0]
                    DNN_answer = action[1:]


                    # new try

                    # select option with highest propability
                    if np.random.random() > epsilon:
                        best_action_index = np.argmax(DNN_answer)
                    else:
                        best_action_index = np.random.randint(0, 40)

                    # algorytm przyblizania najlepszej opcji do najlepszej dostepnej opcji

                    # there are 5 possible set of action
                    optimal_bet = best_action_index * 25

                    if dict_options['check'] and dict_options['raise'] and dict_options['fold'] and dict_options[
                        'all-in']:
                        #print('set 1')
                        if optimal_bet < abs(optimal_bet - min_raise):
                            decision = ['check']
                        elif min_raise <= optimal_bet <= max_raise:
                            decision = ['raise', optimal_bet]
                        else:
                            decision = ['raise', min_raise]

                    # 2 set
                    elif dict_options['call'] and dict_options['raise'] and dict_options['fold'] and dict_options[
                        'all-in']:
                        #print('set 2')
                        if optimal_bet < abs(optimal_bet - call_value):
                            decision = ['fold']
                        elif abs(optimal_bet - call_value) < abs(optimal_bet - min_raise):
                            decision = ['call']
                        elif min_raise <= optimal_bet <= max_raise:
                            decision = ['raise', optimal_bet]
                        else:
                            decision = ['raise', min_raise]

                    # 3 set
                    elif dict_options['call'] and not dict_options['raise'] and dict_options['fold'] and dict_options[
                        'all-in']:
                        #print('set 3')
                        if optimal_bet < abs(call_value - optimal_bet):
                            decision = ['fold']
                        elif abs(optimal_bet - player.stack) < abs(optimal_bet - call_value):
                            decision = ['all-in']
                        else:
                            decision = ['call']

                    # 4 set
                    elif dict_options['check'] and not dict_options['raise'] and dict_options['fold'] and dict_options[
                        'all-in']:
                        #print('set 4')
                        if optimal_bet < abs(optimal_bet - player.stack):
                            decision = ['check']
                        else:
                            decision = ['all-in']

                    # 5 set
                    elif not dict_options['call'] and not dict_options['check'] and not dict_options['raise'] and \
                        dict_options['fold'] and dict_options['all-in']:
                        #print('set 5')
                        if optimal_bet < abs(optimal_bet - player.stack):
                            decision = ['fold']
                        else:
                            decision = ['all-in']

                    # processing action
                    player.action_used = best_action_index


                # Processing of player decision

                if decision[0] == 'raise':
                    chips = int(decision[1])
                    #print(player.name, decision[0], decision[1])
                #else:
                    #print(player.name, decision[0])
                decision = decision[0]

                if decision == 'call':
                    last_action = ['call', 0]
                    chips = max(input_stack_list) - player.input_stack
                    if player.stack > chips:
                        player.drop(chips)
                    else:
                        player.drop(player.stack)
                        player.allin()

                elif decision == 'fold':
                    player.fold()

                elif decision == 'check':
                    last_action = ['check', 0]
                    player.decision = True

                elif decision == 'all-in':
                    last_action = ['raise', player.stack]
                    player.drop(player.stack)
                    for gamer in player_list:
                        # if any of player bets all-in, then each player who not bet all-in and has bet less than
                        # that player all-in, will have to make the decision again
                        if gamer.live and gamer.decision and gamer.input_stack < player.input_stack:
                            gamer.decision = False
                    player.allin()

                elif decision == 'raise':
                    last_action = ['raise', chips]
                    for gamer in player_list:
                        # a czy tutaj nie musi byc tak jak w przypadku all in ze gracze ktorzy mniej postawili
                        if gamer.live and gamer.decision:
                            gamer.decision = False
                    if player.stack > chips:
                        player.drop(chips)
                    else:
                        player.drop(player.stack)
                        player.allin()
                    '''
                    min_raise = 2 * raise_list[0] - raise_list[1]
                    max_raise = player.stack
                    chips = int(input('min raise: ' + str(min_raise) +
                                      ', max raise: ' + str(max_raise) + '. How much :'))
                    while not chips >= min_raise and chips <= max_raise:
                        chips = int(input('min raise: ' + str(min_raise) +
                                          ', max raise: ' + str(max_raise) + '. How much :'))
                    for gamer in player_list:
                        if gamer.live and gamer.decision:
                            gamer.decision = False
                    player.drop(chips)
                    '''
                # for i in player_list:
                #    print(i.name, i.decision)
                # to pozniej wyjebac
                #for playe in player_list:
                #    print(playe.name, ', stack: ', playe.stack, ', bet: ', playe.input_stack)

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
        #print('\n')

    # After auction players who fold or all-in have no decision made until the next round
    for player in player_list:
        player.next_auction()
        if player.live:
            player.decision = False

