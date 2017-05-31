import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from scipy.cluster.vq import kmeans, vq, whiten
from scipy.spatial.distance import cdist
from operator import itemgetter


iris = pd.read_csv('iris.data.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

iris['class'] = pd.Categorical(iris['class']).labels
  
# Plot the sepal length and petal length for each species, and color by species.
print("Petal length and petal width clusters")
plt.scatter(iris['sepal_length'], iris['petal_length'], c=iris['class'])

#Plot attributes other than sepal length and petal length. Does the dataset cluster for some attributes and not for others? Which ones?
plt.scatter(iris['petal_width'], iris['petal_length'], c=iris['class'])

# un data
un = pd.read_csv("un.csv")

print('There are 207 rows in the dataset')

# Determine datatype of each column
print un.dtypes

# How many countries are present in data?
num_countries = len(un['country'])
print('There are %s countries in the dataset' % num_countries)

# need columns 6,7,8,9
data = un.ix[:,:10]
data = data.dropna()

# take data from data for only infant and GDP
thelist = []
for i, row in data.iterrows():
	thelist.append([row['infantMortality'], row['GDPperCapita']])

# 'flatten' the data and make clusters for k=range(1,10)
w = whiten(thelist)
# centers1, dist = kmeans(w, 1)
# centers2, dist = kmeans(w, 2)
# centers3, dist = kmeans(w, 3)
# centers4, dist = kmeans(w, 4)
# centers5, dist = kmeans(w, 5)
# centers6, dist = kmeans(w, 6)
# centers7, dist = kmeans(w, 7)
# centers8, dist = kmeans(w, 8)
# centers9, dist = kmeans(w, 9)
# centers10, dist = kmeans(w, 10)
# TODO: make clusters above with loop
ks = range(1,10)
centers = [kmeans(w,i) for i in ks]

# Just take array of centers from centers for each k
centroids = [cent for (cent,var) in centers]

# calculate distances from data point in w to calculated centroids
distance = [cdist(w, cent, 'euclidean') for cent in centroids]

# create index for distances
cidx = [np.argmin(D, axis=1) for D in distance]

# calculate distances from data point in w to calculated centroids
dist = [np.min(D,axis=1) for D in distance]

# get sum of squares for distances for each k
ssq = [sum(d)/w.shape[0] for d in dist]

# plot line to see where k starts to level out
plt.plot(ks, ssq, 'b*-')

## Infant Mortality & GDP per Cap ##

# assign points to 2 centers (using vq) and plot data with centers
idx2,_ = vq(w, centroids[1])
plt.plot(w[idx2==0,0], w[idx2==0,1], 'ob', 
	 w[idx2==1,0], w[idx2==1,1], 'or')
plt.plot(centroids[1][:,0], centroids[1][:,1], 'sm', markersize=8)

# assign points to 3 centers (using vq) and plot data with centers
idx3,_ = vq(w, centroids[2])
plt.plot(w[idx3==0,0], w[idx3==0,1], 'ob', 
	 w[idx3==1,0], w[idx3==1,1], 'or',
	 w[idx3==2,0], w[idx3==2,1], 'og')
plt.plot(centroids[2][:,0], centroids[2][:,1], 'sg', markersize=8)

# assign points to 5 centers (using vq) and plot data with centers
idx5,_ = vq(w, centroids[4])
plt.plot(w[idx5==0,0], w[idx5==0,1], 'ob', 
	 w[idx5==1,0], w[idx5==1,1], 'or',
	 w[idx5==2,0], w[idx5==2,1], 'og',
	 w[idx5==3,0], w[idx5==3,1], 'om',
	 w[idx5==4,0], w[idx5==4,1], 'oy')
plt.plot(centroids[4][:,0], centroids[4][:,1], 'sg', markersize=12)

## Male Life Expectancy & GDP per Cap ##
thelist = []
for i, row in data.iterrows():
	thelist.append([row['GDPperCapita'], row['lifeMale']])

w = whiten(thelist)
ks = range(1,10)
centers = [kmeans(w,i) for i in ks]
centroids = [cent for (cent,var) in centers]

# assign points to 3 centers (using vq) and plot data with centers
idx3,_ = vq(w, centroids[2])
plt.plot(w[idx3==0,0], w[idx3==0,1], 'ob', 
	 w[idx3==1,0], w[idx3==1,1], 'or',
	 w[idx3==2,0], w[idx3==2,1], 'og')
plt.plot(centroids[2][:,0], centroids[2][:,1], 'sg', markersize=8)

# see what countrys are in what clusters
countrylist = []
for i, row in data.iterrows():
	countrylist.append(row['country'])

final = []
for i in range(len(countrylist)):
	final.append([countrylist[i], idx3[i]])

sorted(final, itemgetter(1))