#Python 3 code, compiled in JupyterHub

import random
# the following function generates head and tail randomly with prob of a head being 0.6
def flip(p):
    return 'H' if random.random() < p else 'T'
# a short experiment if we are getting right results!
N = 300
flips = [flip(0.6) for i in range(N)]
float(flips.count('H'))/N



# money is the parameter we have to choose and experiment with i.e. what fixed bet should be everytime
# ct keeps track of how many times we are getting to play the coin toss
def fixed_bet(money):
    amt = 25
    ct = 0
    for i in range(300):
        if amt <= 0: break # this is getting busted!
        if amt >= 250: break # limit reached!
        if flip(0.6) == "H":
            print(ct)
            ct += 1
            amt += money # updating money
        else:
            print(ct)
            ct += 1
            amt -= money # updating money
    return amt
# now we play with the parameter money to find what works best and put it in ans
# the follwoing experiment is done with all the strategies.
ans = []
for i in range(1,25):
    if fixed_bet(i) > 249: ans.append(i) # replace 249 with any number in [1,250] to check if you can cross that
ans


def martingale_bet(starter):
    amt = 25
    ct = 0
    for i in range(300):
        if amt <= 0: break
        if amt >= 250: break
        if flip(0.6) == "H":
            print(ct)
            amt += starter
            ct += 1
        else:
            print(ct)
            amt -= starter
            starter = 2 * starter # if we lose, we double our bet to be back on track on first win!
            ct += 1
    return amt

# experiment with the starting bet
ans = []
for i in range(1,25):
    if martingale_bet(i) > 249: ans.append(i) # replace 249 with whatever!
ans


def proportional_bet(frac):
    amt = 25
    ct = 0
    for i in range(300):
        if amt <= 0: break
        if amt >= 250: break
        if flip(0.6) == "H":
            amt = amt * (1+frac)
            print(ct)
            ct += 1
        else:
            amt = amt * (1-frac)
            print(ct)
            ct += 1
    return amt

# experiment with the frac
ans = []
fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # we can put any value between 0 and 1 in this list, but we already know 0.2 works best!
for frac in fracs:
    if proportional_bet(frac) > 249: # replace 249 with whatever! we have tried with 200,225,240 and 250
        ans.append(frac)
ans
