
dict_options = {'fold': True, 'all-in': True, 'call': False, 'check': True, 'raise': True}


# optimal bet  = index_action * 25
optimal_bet = 180
call_value = 100
min_raise = 150
max_raise = 600
playerstack = 600


# 1 set

if dict_options['check'] and dict_options['raise'] and dict_options['fold'] and dict_options['all-in']:
    print('set 1')
    if optimal_bet < abs(optimal_bet - min_raise):
        decision = ['check']
    elif min_raise <= optimal_bet <= max_raise:
        decision = ['raise', optimal_bet]
    else:
        decision = ['raise', min_raise]

# 2 set
elif dict_options['call'] and dict_options['raise'] and dict_options['fold'] and dict_options['all-in']:
    print('set 2')
    if optimal_bet < abs(optimal_bet - call_value):
        decision = ['fold']
    elif abs(optimal_bet - call_value) < abs(optimal_bet - min_raise):
        decision = ['call']
    elif min_raise <= optimal_bet <= max_raise:
        decision = ['raise', optimal_bet]
    else:
        decision = ['raise', min_raise]

# 3 set
elif dict_options['call'] and not dict_options['raise'] and dict_options['fold'] and dict_options['all-in']:
    print('set 3')
    if optimal_bet < abs(call_value - optimal_bet):
        decision = ['fold']
    elif abs(optimal_bet - playerstack) < abs(optimal_bet - call_value):
        decision = ['all-in']
    else:
        decision = ['call']

# 4 set
elif dict_options['check'] and not dict_options['raise'] and dict_options['fold'] and dict_options['all-in']:
    print('set 4')
    if optimal_bet < abs(optimal_bet - playerstack):
        decision = ['check']
    else:
        decision = ['all-in']

# 5 set
elif not dict_options['call'] and not dict_options['check'] and not dict_options['raise'] and dict_options['fold'] and dict_options['all-in']:
    print('set 5')
    if optimal_bet < abs(optimal_bet - playerstack):
        decision = ['fold']
    else:
        decision = ['all-in']

print(decision)