import time

from game.deepAI import probability_win


own_cards = ['AD', '2C']
n_player = 2
start = time.time()
p_win, p_tie = probability_win(own_cards, n_player)
stop = time.time()

print('P win: {}%, P tie: {}%, time: {}'.format(p_win * 100, p_tie * 100, stop - start))
