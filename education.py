from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import sqlite3 as lite
import statsmodels.formula.api as smf
import math

# store url for school years
url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/demographic/products/socind/education.htm"

# get the html
r = requests.get(url)

# parse the html content with bs
soup = BeautifulSoup(r.content)


mylist = soup.findAll('tr', attrs=('class', 'tcont'))
mylist = mylist[:93]

#country_name, year, school years, male , female 
countries = []
for item in mylist:
	countries.append([item.contents[1].string,
		          item.contents[3].string,
                  item.contents[9].string,
		          item.contents[15].string,
		          item.contents[21].string])

# convert data to pandas dataframe and define column names
df = pd.DataFrame(countries)
df.columns = ['Country', 'DataYear', 'TotalYears', 'MaleYears', 'FemaleYears']

# convert school years to integers
df['TotalYears'] = df['MaleYears'].map(lambda x: int(x))
df['MaleYears'] = df['MaleYears'].map(lambda x: int(x))
df['FemaleYears'] = df['FemaleYears'].map(lambda x: int(x))

print("The city mean years are:")
mean = df.mean()
print mean

print("The city mean years are:")
median = df.median()
print median

max = df.max()
print("The maximum years is")
print max

min = df.min()
print("The minimum years is")
print min

con = lite.connect('education.db')
with con:  
    cur = con.cursor()
    df.to_sql("education_years", con, if_exists="replace")
    cur.execute("DROP TABLE IF EXISTS gdp")
    cur.execute('CREATE TABLE gdp (country_name text, _1999 integer, _2000 integer, _2001 integer, _2002 integer, _2003 integer, _2004 integer, _2005 integer, _2006 integer, _2007 integer, _2008 integer, _2009 integer, _2010 integer)')
        
with open('API_NY.GDP.MKTP.CD_DS2_en_csv_v2.csv','rU') as inputFile:
    next(inputFile)
    next(inputFile)
    next(inputFile)
    next(inputFile)
    header = next(inputFile)
    inputReader = csv.reader(inputFile)
    for line in inputReader:
        cur.execute('INSERT INTO gdp (country_name, _1999, _2000, _2001, _2002, _2003, _2004, _2005, _2006, _2007, _2008, _2009, _2010) VALUES ("' + line[0] + '","' + '","'.join(line[42:-8]) + '");')
        cur.execute("SELECT country_name, TotalYears, _2000, _2005, _2010 FROM education_years INNER JOIN gdp ON Country = country_name")
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        gdp_df = pd.DataFrame(rows, columns=cols)



est = smf.ols(formula='TotalYears ~ _2010', data=gdp_df).fit()
print(est.summary())


