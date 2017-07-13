import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
def create_random_data():
    n = 500000
    d = 500000
    choice =np.random.choice(range(2),n, p=[0.9,0.1])
    print choice
    data = []
    for i in xrange(n):
        print i
        if choice[i] ==0:
            data.append(np.random.choice(xrange(700)))
        else :
            data.append(np.random.choice(xrange(d)))


    print data
    #data = np.random.choice(range(10), 10000, p=[0.15, 0.15, 0.2, 0.1, 0.37, 0.006, 0.006, 0.006, 0.006, 0.006])
    #hist = Counter(data)
    #print hist
    #labels, values = zip(*Counter(data).items())
    #indexes = np.arange(len(labels))
    #width = 1
    #plt.bar(indexes, values, width)
    #plt.xticks(indexes + width * 0.5, labels)
    #plt.show()
    file = open('n=500000hh=700d=500000.txt', 'w')
    for item in data:
        print>> file, item




create_random_data()





