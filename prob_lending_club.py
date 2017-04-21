#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:46:50 2017

@author: amy
"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')
loansData.dropna(inplace=True)
df = pd.DataFrame(loansData)

print("From these charts, we can see that requests for funding are typically funded. The most common requests are somewhere between $5,000 and $10,000.")

print("This is a Box Plot of the amount funded by investors.")
loansData.boxplot(column='Amount.Funded.By.Investors')

plt.show()

print("This is a Box Plot of the amount requested.")
loansData.boxplot(column='Amount.Requested')

plt.show()

print("This is a histogram of the amount funded by investors.")
loansData.hist(column='Amount.Funded.By.Investors')

plt.show()

print("This is a histogram of the amount requested.")
loansData.hist(column='Amount.Requested')

plt.show()

print("This is QQ-plot of the amount funded by investors and amount requested.")
plt.figure()
graph = stats.probplot(loansData['Amount.Funded.By.Investors'], dist="norm", plot=plt)


plt.figure()
graph = stats.probplot(loansData['Amount.Requested'], dist="norm", plot=plt)



