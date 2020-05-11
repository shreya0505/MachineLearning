import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = './Datasets/Data8.csv'
dataset = pd.read_csv(filename, header = None)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

results = list(rules)
for i in range(len(results)):
    print(results[i],'\n')

