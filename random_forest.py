import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

newlines = []
f = open("features.txt", "r")
lines = f.read().split('\n')
for x in lines:
    newlines.append(x.replace('-', '').replace('(', '').replace(')', '').replace(',', '').replace('Body', '').replace('Mag', '').replace('Mean', 'mean').replace('STD', 'std').replace(' ', ''))
  
f.close()

X_train = pd.read_csv('X_train.txt', sep=' ', header=None, names = newlines)
y_train = pd.read_csv('y_train.txt')

#removes last blank column in X_train
X_train.drop("", inplace=True, axis=1)

#Plot a histogram of Body Acceleration Magnitude     
X_train.hist(column='tAccmean1')           
plt.show()

#split the x_train data into training, testing, and validation sets
x_train, x_validate, x_test = np.split(X_train.sample(frac=1), [int(.6*len(X_train)), int(.8*len(X_train))])

#split the y_train data into training, testing, and validation sets
y_train, y_validate, y_test = np.split(y_train.sample(frac=1), [int(.6*len(y_train)), int(.8*len(y_train))])



#fit a random forest classifier with 500 estimators to your training set. Can change parameters to change how well model fits data set. Can validate it with x validate and y validate. 
est = RandomForestClassifier(n_estimators=500)


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
print confusion_matrix(y_test, y_test_predict)

#prepare a range of parameter values to test
#max_features = np.array([None, "sqrt", "log2"])
#grid = GridSearchCV(estimator=est, param_grid=dict(max_features=max_features))
#grid.fit(x_train, y_train)
#print(grid)
# summarize the results of the grid search
#print(grid.best_score_)
#print(grid.best_estimator_.max_features)

#prepare a range of parameter values to test
parameters = {'max_features':(None, 'sqrt', 'log2'), 'max_depth':('integer', None), 'min_samples_split':('int', 'float')}
grid = GridSearchCV(estimator=est, param_grid=dict(parameters=parameters))
grid.fit(x_train, y_train)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.parameters)



