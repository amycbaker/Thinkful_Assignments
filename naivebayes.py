import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


#reads features data, using space as a delimiter and creating column name feature_names
df = pd.read_csv('ideal_weight.csv', index_col=[0], delimiter=',')


#cleans data per lesson instructions using string replace
df.columns = df.columns.str.replace('\'','')
df['sex'] = df['sex'].str.replace('\'','')

print("Actual")
plt.figure()
p = df['actual'].hist()
plt.show()

print("Ideal")
plt.figure()
p = df['ideal'].hist()
plt.show()

print("Difference")
plt.figure()
p = df['diff'].hist()
plt.show()

df['sex'] = df['sex'].map(lambda x: 0 if x == 'Male' else 1)

num_women = sum(df['sex'])
num_men = len(df['sex']) - num_women

print('There are %s women and %s men.' %(num_women, num_men))

train, test = train_test_split(df, test_size = 0.3)


gnb = GaussianNB()
data = df[['actual','ideal','diff']]
target = df['sex']
model = gnb.fit(data, target)
y_pred = model.predict(data)
print("Number of mislabeled points out of a total %d points: %d" %(data.shape[0], (target != y_pred).sum()))

# Predict the sex for an actual weight of 145, an ideal weight of 160, and a diff of -15." 

d = {'actual': 145, 'ideal': 160, 'diff': -15}
df = pd.DataFrame(data=d, index=[1])
df = df[['actual','ideal', 'diff']]
pred = model.predict(df)
print pred

d = {'actual': 160, 'ideal': 145, 'diff': -15}
df = pd.DataFrame(data=d, index=[1])
df = df[['actual','ideal', 'diff']]
pred = model.predict(df)
print pred

