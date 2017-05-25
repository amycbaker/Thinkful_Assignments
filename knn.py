from sklearn import datasets
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
import random
import math


df = pd.read_csv('irisdata.csv', names=['sepal_length', 'sepal_width', 'pedal_length', 'pedal_width', 'class'])


# Plot sepal length against sepal width.
plt.scatter(df.sepal_length, df.sepal_width)

#generate a random point programatically
random.seed()
pt = df.iloc[random.choice(df.index.tolist())]
pt['sepal_length']

def dist_from_pt(p):
    return math.sqrt(((pt.sepal_length - p.sepal_length) ** 2) + ((pt.sepal_width - p.sepal_width) ** 2))

df['dist_from_pt'] = df[['sepal_length', 'sepal_width']].apply(func=dist_from_pt, axis=1)
df.head()


# Look at 10 nearest neighbors.
df_sorted = df.sort_values(by='dist_from_pt', ascending=True)
df_sorted[0:10]


df_sorted['class'][0:10].value_counts().index[0]


def knn(k):
    return df_sorted['class'][0:k].value_counts().index[0]

print(knn(50))