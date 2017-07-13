import numpy as np
import math
import numpy as np
import itertools
import random
from collections import Counter

def randomized_resopnse(bit_x, epsilon):
    threshold = float(np.exp(epsilon)) / float(np.exp(epsilon) + 1)
    # if the random float falls in the range [0, e^epsilon\e^epsilon)+1] return bit_x otherwise flip bit
    if random.random() <= threshold:
        return bit_x
    else:
        return -bit_x

def create_random_Hash(T,data):
    hash_function = {}
    for value in data:
        if value not in hash_function:
            hash_function[value] = random.randint(0,T-1)
    return hash_function


d = 500000
T=700
epsilon = 2.0
data = lines = [line.rstrip('\n') for line in open('test.txt')]
n= len(data)
hash = create_random_Hash(T,data)
sums = {t:0 for t in range(T)}
print Counter(data)
for t in range(T):
    for i in range(n):
        z = random.choice([-1, 1])
        if hash[data[i]] == t:
            sums[t]+= z*randomized_resopnse(z,epsilon)
        else:
            sums[t]+= z
print sums





