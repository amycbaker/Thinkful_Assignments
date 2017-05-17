import requests
import datetime
import sqlite3 as lite
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt




con = lite.connect('weather.db')
cur = con.cursor()

df = pd.read_sql_query("SELECT * FROM daily_temp", con, index_col = "day_of_reading")

# Close connection for good practice
con.close()


#What's the range of temperatures for each city? 
range = df.max() - df.min()
print("The temperature ranges for each city are:")
print diff

#What is the mean temperature for each city? 
print("The city mean temperatures are:")
mean = df.mean()
print mean

print("The variance in temperatures are:")
variance = df.var()
print variance

ax = df[['Atlanta']].plot(kind='bar', title ="Atlanta Daily Temperatures", figsize=(15, 10), fontsize=12)
ax.set_xlabel("City",fontsize=12)
ax.set_ylabel("Temperature",fontsize=12) 

ax = df[['Austin']].plot(kind='bar', title ="Austin Daily Temperatures", figsize=(15, 10), fontsize=12)
ax.set_xlabel("City",fontsize=12)
ax.set_ylabel("Temperature",fontsize=12) 

ax = df[['Boston']].plot(kind='bar', title ="Boston Daily Temperatures", figsize=(15, 10), fontsize=12)
ax.set_xlabel("City",fontsize=12)
ax.set_ylabel("Temperature",fontsize=12) 

ax = df[['Chicago']].plot(kind='bar', title ="Chicago Daily Temperatures", figsize=(15, 10), fontsize=12)
ax.set_xlabel("City",fontsize=12)
ax.set_ylabel("Temperature",fontsize=12) 

ax = df[['Cleveland']].plot(kind='bar', title ="Cleveland Daily Temperatures", figsize=(15, 10), fontsize=12)
ax.set_xlabel("City",fontsize=12)
ax.set_ylabel("Temperature",fontsize=12) 

print("Atlanta and Austin have warmer climates with less variation, while Boston, Chicago, and Cleveland experience colder days with more variation in daily temperature.")