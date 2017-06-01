from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split, cross_val_score
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

iris = datasets.load_iris()
df_iris = pd.DataFrame(iris.data, columns = iris.feature_names)
df_iris['Species'] = iris.target


 #Use the cross_validation.train_test_split() helper function to split the Iris dataset into training and test sets, holding out 40% of the data for testing. 
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

  #How many points do you have in your training set? In your test set?
X_test_points = len(X_test)
X_training_points = len(X_train)
print("There are %s points in the test set and %s points in the training set." %(X_test_points, X_training_points))

#Fit a linear Support Vector Classifier to the training set and evaluate its performance on the test set. 
svc = svm.SVC(kernel = 'linear')
svc.fit(X_train, y_train)
predicted = svc.predict(X_test)
expected = X_test

print 'For each prediction, did we predict correctly?'
for i in range(len(predicted)):
	if predicted[i] == y_test[i]:
		print 'Prediction correct'
	else:
		print 'Prediction Incorrect'
        
        
#What is the score? 
print 'Getting accuracy score...'
print accuracy_score(y_test, predicted)

kf = KFold(n_splits=5)
mse = 0
for train, test in kf.split(df_iris):
    svc = svm.SVC(kernel = 'linear')
    svc.fit(X_train, y_train)
    predicted = svc.predict(X_test)
    expected = X_test
    print("Mean Squared Error")
    print(mean_squared_error(y_test, predicted))
    mse = 0
    mse = mse + mean_squared_error(y_test, predicted)
    print("Mean Absolute Error")
    print(mean_absolute_error(y_test, predicted))
    mae = 0
    mae = mae + mean_squared_error(y_test, predicted)
    print("R Squared Score")
    print(r2_score(y_test, predicted))
    avr = 0
    avr = avr + r2_score(y_test, predicted)
    
print("The average Mean Squared Error is %s" % mse)
print("The average Mean Squared Error is %s" % mae)
print("The average r squared value is %s" % avr)
