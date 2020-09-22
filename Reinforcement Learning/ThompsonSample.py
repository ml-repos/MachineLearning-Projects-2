import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('ads.csv')

import random
N = 10000
d = 10
n_rewards_1 = [0]*d
n_rewards_0 = [0]*d
ads_select = []
total_reward = 0
for n in range(0, N):
    ad, max_random = 0,0
    for i in range(0, d):
        random_beta = random.betavariate(n_rewards_1[i]+1, n_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_select.append(ad)
    reward = dataset.values[n, ad]
    if reward==1:
        n_rewards_1[ad] = n_rewards_1[ad] + 1
    else:
        n_rewards_0[ad] = n_rewards_0[ad] + 1
    total_reward = total_reward + reward


plt.hist(ads_select)
plt.show()