import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mylist = [1, 1, 4, 4, 5, 6, 7, 9, 9, 9, 9, 10, 12, 13, 13, 14]

c = collections.Counter(mylist)

print(c)

# calculate the number of instances in the list
count_sum = sum(c.values())

for k,v in c.items():
  print("The frequency of number %s is %s" % (k, float(v) / count_sum))
  
print("This is a boxplot of My List")
plt.boxplot(mylist)
plt.show()

print("This is a histogram of My List")
plt.hist(mylist, histtype='bar')
plt.show()

print("This is QQ plot of My List"p)
plt.figure()
graph1 = stats.probplot(mylist, dist="norm", plot=plt)
