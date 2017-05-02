# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:26:33 2017

@author: ab1
"""

import pandas as pd #imports pandas
import statsmodels.api as sm #imports statsmodel
import numpy as np

loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')
#imports initial dataset csv
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(x[:3]))
#cleans FICO.Score column, removing second FICO score and changing it into an integer. 
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
#cleans Interest.Rate column, removing %, changing it to a decimal. 
loansData['Interest.Rate'] = cleanInterestRate
#changing Interest.Rate column to equal cleanInterestrate calculation in previous line


loansData.to_csv('loansData_clean.csv', header=True, index=False)
#loads the previously cleaned data into a new CSV
df = pd.read_csv('loansData_clean.csv', index_col=False) 
#loads the CSV data in a pandas dataframe
df['IR_TF'] = df['Interest.Rate'].map(lambda x: 0 if x < .12 else 1)
#creates a new column where if the interest rate is below 12%, then a value of 0 is assigned, otherwise a value of 1 is assigned.
df['Intercept'] = 1.0
#creates an intercept column with a constant value of 1


ind_vars = ['Intercept', 'FICO.Score', 'Amount.Requested']
#creates a variable with all of the independent variables
 
logit = sm.Logit(df['IR_TF'], df[ind_vars])
#This is where the error starts.
#Define the logistic regression model.

result = logit.fit()
#Fit the model

coeff = result.params
print(coeff)
#Get the fitted coefficients from the results.

def logistic_function(FicoScore, LoanAmount, coeff):
    p = 1 / (1 + np.exp(-(coeff[0] + (coeff[1]*FicoScore) + (coeff[2]*LoanAmount))))
    return p

def pred(FicoScore, LoanAmount, coeff):
    p = logistic_function(FicoScore, LoanAmount, coeff)
    if p >= 0.7:
        print "Loan approved"
    else:
        print "Loan denied"
        
pred(720, 10000, coeff)


 





