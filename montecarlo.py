#'''This script demonstrates simulations of coin flipping'''
import random
import matplotlib.pyplot as plt
import numpy as np


# let's create a fair coin object that can be flipped:

class Coin(object):
    '''this is a simple fair coin, can be pseudorandomly flipped'''
    sides = ('heads', 'tails')
    last_result = None

    def flip(self):
        '''call coin.flip() to flip the coin and record it as the last result'''
        self.last_result = result = random.choice(self.sides)
        return result

# let's create some auxilliary functions to manipulate the coins:

def create_coins(number):
    '''create a list of a number of coin objects'''
    return [Coin() for _ in xrange(number)]

def flip_coins(coins):
    '''side effect function, modifies object in place, returns None'''
    for coin in coins:
        coin.flip()

def count_heads(flipped_coins):
    return sum(coin.last_result == 'heads' for coin in flipped_coins)

def count_tails(flipped_coins):
    return sum(coin.last_result == 'tails' for coin in flipped_coins)

coins_list = []

def main():
    coins = create_coins(1000)
    for i in xrange(100):
        flip_coins(coins)
        coins_list.append(count_heads(coins))
        print(count_heads(coins))

if __name__ == '__main__':
    main()

plt.hist(coins_list)
plt.show()
#plt.hist(max_list)
#plt.show()
#plt.hist(min_list)
#plt.show()

#Modify the above program to generate trials of a normal variable. 
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

#Verify the mean and the variance:


abs(mu - np.mean(s)) < 0.01
True

abs(sigma - np.std(s, ddof=1)) < 0.01
True

#Display the histogram of the samples, along with the probability density function:

count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
          linewidth=2, color='r')
plt.show()

