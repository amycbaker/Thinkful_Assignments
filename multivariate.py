# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:26:33 2017

@author: ab1
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


loansData = pd.read_csv('https://raw.githubusercontent.com/Thinkful-Ed/curric-data-001-data-sets/master/loans/loansData.csv')
loansData.dropna(inplace=True)

cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
loansData['Interest.Rate'] = cleanInterestRate
intrate = loansData['Interest.Rate']
income = loansData['Monthly.Income']
home_ownership = loansData['Home.Ownership'].map(lambda x: 1 if x == "MORTGAGE" else 0)


#Use income (annual_inc) to model interest rates (int_rate).


#The dependent variable 
y = np.matrix(intrate).transpose()
#The independent variable shaped as columns
x1 = np.matrix(income).transpose()
# The independent variable shaped as a column
x = np.column_stack([x1])

#linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print f.summary()

#The dependent variable 
y = np.matrix(intrate).transpose()
#The independent variable shaped as columns
x1 = np.matrix(income).transpose()
x2 = np.matrix(home_ownership).transpose()
# The independent variable shaped as a column
x = np.column_stack([x1,x2])

#linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print f.summary()

#testing for interaction of home ownership and income
est = smf.ols(formula="intrate ~ income * home_ownership", data=loansData).fit()

print(est.summary())


