import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')

loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(x[:3]))

cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
loansData['Interest.Rate'] = cleanInterestRate
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

#The dependent variable 

#print f.summary()
kf = KFold(n_splits=10)
mse = 0
for train, test in kf.split(loansData):
#     print("%s %s" % (train, test))
    y = np.matrix(intrate.iloc[train]).transpose()
#     The independent variable shaped as columns
    x1 = np.matrix(fico.iloc[train]).transpose()
    x2 = np.matrix(loanamt.iloc[train]).transpose()

    x = np.column_stack([x1,x2])

    X = sm.add_constant(x)
    model = sm.OLS(y,X)
    f = model.fit()

    x1_test = np.matrix(fico.iloc[test]).transpose()
    x2_test = np.matrix(loanamt.iloc[test]).transpose()

    x_test = np.column_stack([x1_test,x2_test])
    X_test = sm.add_constant(x_test)
    
    y_pred = f.predict(X_test)
    
    y_test = np.matrix(intrate.iloc[test]).transpose()
    print("Mean Squared Error")
    print(mean_squared_error(y_test, y_pred))
    mse = mse + mean_squared_error(y_test, y_pred)
    print("Mean Absolute Error")
    print(mean_absolute_error(y_test, y_pred))
    mae = mae + mean_squared_error(y_test, y_pred)
    print("R Squared Score")
    print(r2_score(y_test, y_pred))
    avr = avr + r2_score(y_test, y_pred)
    
    
print("Average MSE")
print(mse / 10)

print("Average MAE")
print(mae / 10)

print("Average R Squared")
print(avr / 10)

