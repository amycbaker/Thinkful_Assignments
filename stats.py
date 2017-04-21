import pandas as pd
from scipy import stats

data = '''Region,Alcohol,Tobacco
North,6.47,4.03
Yorkshire,6.13,3.76
Northeast,6.19,3.77
East Midlands,4.89,3.34
West Midlands,5.63,3.47
East Anglia,4.52,2.92
Southeast,5.89,3.20
Southwest,4.79,2.71
Wales,5.27,3.53
Scotland,6.08,4.51
Northern Ireland,4.02,4.56'''

data =  data.splitlines()

data = [i.split(',') for i in data]

column_names = data[0] #this is the first row
data_rows = data[1::] #these are all of the following rows of data
df = pd.DataFrame(data_rows, columns=column_names)

df['Alcohol'] = df['Alcohol'].astype(float)
df['Tobacco'] = df['Tobacco'].astype(float)



#range
range = max(df['Alcohol']) - min(df['Alcohol'])
print("The range for Alcohol is %s" % range) 

#standard deviation
sd = df['Alcohol'].std()
print("The standard deviation for Alcohol is %s" % sd) 

#variance
vari = df['Alcohol'].var()
print("The variance for Alcohol is %s" % vari)

alc_mean = df['Alcohol'].mean()
print("The mean for Alcohol is %s" % alc_mean)

alc_median = df['Alcohol'].median()
print("The median for Alcohol is %s" % alc_median)

#range
range = max(df['Tobacco']) - min(df['Tobacco'])
print("The range for Tobacco is %s" % range)

#standard deviation
sd = df['Tobacco'].std()
print("The standard deviation for Tobacco is %s" % sd) 

#variance
vari = df['Tobacco'].var()
print("The variance for Tobacco is %s" % vari)

tob_mean = df['Tobacco'].mean()
print("The mean for Tobacco is %s" % tob_mean)

tob_median = df['Tobacco'].median()
print("The median for Tobacco is %s" % tob_median)

tob_array = stats.mode(df['Tobacco'])
tob_mode = tob_array[0]
print("The mode for Tobacco is %s" % tob_mode)


alc_array = stats.mode(df['Alcohol'])
alc_mode = alc_array[0]
print("The mode for Alcohol is %s" % alc_mode)







 