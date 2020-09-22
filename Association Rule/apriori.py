import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv("market.csv",header=None)

# transaction = []
# for i in range(0, 7501):
#     transaction.append([str(dataset.values[i,j]) for j in range(0, 20)])
# with open('./market.pkl', 'wb') as f:
#     pickle.dump(transaction, f)

with open('./market.pkl', 'rb') as f:
    transaction =pickle.load(f)


from apyori import apriori
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

results = list(rules)
