import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv('SIF.csv')
df.dropna(inplace=True)
#fills blank interupted customer fields with zeroes
df['IntCust'].fillna(0)


SIF = pd.read_csv('SIFscatter.csv')
SIF.dropna(inplace=True)
#Plots a scatterplot matrix of multiple variables to show the relationship of each variable to each of the others. 
a = pd.scatter_matrix(SIF, alpha=0.05, figsize=(10,10), diagonal='hist')

#Plots a scatter of Voltage vs. Interrupted Customers
plt.scatter(df['Voltage'],df['IntCust'])
plt.show()


plt.hist(df['IntCust']) 
plt.title("Number of Interrupted Customers")
plt.show()

plt.hist(df['IntCust']) 
axes = plt.gca()
axes.set_ylim([0, 10])
plt.title("Number of Interrupted Customers - modified y-axis")
plt.show()


plt.hist(df['IntHours']) 
axes = plt.gca()
plt.title("Number of Interruption Hours")
plt.show()

plt.hist(df['IntHours']) 
axes = plt.gca()
axes.set_ylim([0, 10])
plt.title("Number of Interruption Hours - modified y-axis")
plt.show()


#Linear regreationt
Voltage = df['Voltage']
IntCust = df['IntCust']


#Linear regression model
model = sm.OLS(IntCust, Voltage)
results = model.fit()

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(results, 0, ax=ax)
ax.set_ylabel("Interrupted Customers")
ax.set_xlabel("Voltage")
ax.set_title("Linear Regression")
plt.show()



#OLS regression results
print results.summary()

#From these results we can see that there is a very small coeeffient of .0010. The P value is zero, but we also have a 
#very poor R-squared result of .038, showing that the model does not fit the data very well. 
#this is not a surprise given that the scatterplots did not show a strong correlation between these two variables. 


X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

#split the x_train data into training, testing, and validation sets
x_train, x_validate, x_test = np.split(X_train.sample(frac=1, random_state=1), [int(.6*len(X_train)), int(.8*len(X_train))])

#split the y_train data into training, testing, and validation sets
y_train, y_validate, y_test = np.split(y_train.sample(frac=1, random_state=1), [int(.6*len(y_train)), int(.8*len(y_train))])



#fit a random forest classifier with 500 estimators to your training set. Can change parameters to change how well model fits data set. Can validate it with x validate and y validate. 
est = RandomForestClassifier(n_estimators=500, min_samples_leaf=70)


#fit (X,y)
y_train = y_train.ix[:,0]
est.fit(x_train, y_train)


#run predictions
y_test_predict = est.predict(x_test)
y_validate_predict = est.predict(x_validate)



#rank the features by their importance score. What are the top 10? 
print ("Feature Importances")
print est.feature_importances_

importances = est.feature_importances_
std = np.std([tree.feature_importances_ for tree in est.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the rop 10 feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

#compare to real y_test and real y_validate with the following:
#mean accuracy score on the validation and test sets
print("Mean accuracy score test set")
print est.score(x_test, y_test)

print("Mean accuracy score validation set")
print est.score(x_validate, y_validate)

#precision and recall score on the test set? metrics.f1_score(y_true, y_predict)
print("Precision and recall score")
f1_score = metrics.f1_score(y_test, y_test_predict, average='micro')
print f1_score

# confusion matrix is a visual representation
conf = confusion_matrix(y_test, y_test_predict)
print conf
plt.imshow(conf, cmap='binary', interpolation='None')
plt.show()


#1. feature 45 (Ambulance) (0.230049)
#2. feature 9 (Service Interruption) (0.152075)
#3. feature 1 (Fatality) (0.092873)
#4. feature 2 (Injury) (0.065420)
#5. feature 44 (Voltage) (0.056642)
#6. feature 42 (Cause - Working Overhead) (0.056259)
#7. feature 10 (Number of Interrupted Customers) (0.050908)
#8. feature 43 (Cause - Working Underground)(0.048218)
#9. feature 7 (Estimated other damage in dollars) (0.041581)
#10. feature 8 (Total damage in dollars) (0.038556)



#changing minimum sample leaf size to 70 improved mean accuracy score of the test set  and the validation set . 


y_test_df = pd.DataFrame(y_test)
y_test_predict_df = pd.DataFrame(y_test_predict)

y_test_df.to_csv('y_test_df.csv')
y_test_predict_df.to_csv('y_test_predict_df.csv')



