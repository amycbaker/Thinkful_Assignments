#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:48:26 2017

@author: amy
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('/Users/amy/Documents/Thinkful/Thinkful_Assignments/LoanStats3b.csv', header=1, low_memory=False)

df['issue_d_format'] = pd.to_datetime(df['issue_d'])
dfts = df.set_index('issue_d_format')
year_month_summary = dfts.groupby(lambda x: x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

plt.figure()
p = loan_count_summary.hist()
plt.show()

sm.graphics.tsa.plot_acf(loan_count_summary)
sm.graphics.tsa.plot_pacf(loan_count_summary) 