import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('ads.csv')

import math
N = 10000
d = 10
n_selection = [0]*d
sum_reward = [0]*d
ads_select = []
total_reward = 0
for n in range(0, N):
    ad, max_ub = 0,0
    for i in range(0, d):
        if n_selection[i] > 0:
            average_reward = sum_reward[i] / n_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/n_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_ub:
            max_ub = upper_bound
            ad = i
    ads_select.append(ad)
    n_selection[ad] = n_selection[ad] + 1
    reward = dataset.values[n, ad]
    sum_reward[ad] = sum_reward[ad] + reward
    total_reward = total_reward + reward


plt.hist(ads_select)
plt.show()