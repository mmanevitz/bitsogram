
import math
import numpy as np
import itertools
import random
from Crypto.Cipher import AES
from Crypto import Random
from user_side import  user_Node
import bitstring
from config import config
import tabels


def int_to_binary(number,length):
    return format(number, '0'+str(length)+'b')

def binary_to_int(value):
    return int(value, 2)


#devides the users into numberOfsubsets disjoint groups
def random_partition(users,numberOfsubsets):
    partition = {i:[] for i in range(len(users))}
    for user in users:
        partition[user.index] = random.randint(0,numberOfsubsets)
    print partition



def createRandomMatrix(T,n):

    return np.random.choice([-1,1], size=(T,n))

def create_random_Hash(T,users):
    hash_function = {}
    for user in users:
        if user.value not in hash_function:
            hash_function[user.value] = random.randint(0,T-1)
    return hash_function

#
# def random_hashtogram(users, R, T,epsilon):
#     Hash_functions = [create_random_Hash(T,users) for i in range(R)]
#     ratio = float(np.exp(epsilon+1))  / float(np.exp(epsilon) -1)
#     n =len(users)
#     A = np.zeros(R,T)
#     responses =
#     Z = createRandomMatrix(T,n)
#     for j in n:
#         partition_number = random.randint(range(R))
#         users[j].random_Response(Z, create_random_Hash(r))
#         for r,t in itertools.product(xrange(R),xrange(T)):
#             A(r,t) = ratio
#
#






def randomized_resopnse(bit_x, epsilon):
    threshold = float(np.exp(epsilon)) / float(np.exp(epsilon) + 1)
    # if the random float falls in the range [0, e^epsilon\e^epsilon)+1] return bit_x otherwise flip bit
    if random.random() <= threshold:
        return bit_x
    else:
        return -bit_x



def create_random_hash_functions():
    print "hello"

def Hashtogram_aggregate():
    print "hello"

def random_partition(n,number_of_subsets):
    partition ={j: [] for j in range(number_of_subsets)}
    for i in range(n):
        set_number = random.randint(0,number_of_subsets-1)
        partition[set_number].append(i)
    return partition


# def ExplicitHist1(d,T, partition,users, hash_func, epsilon):
#
#     np.zeroes()
#     number_of_subsets = len(partition)
#     for l in range(number_of_subsets):
#         user_indexes = partition[l]
#         numer_of_users_in_partition = len(user_indexes)
#         y = []
#         Z =create_random_matrix(2*T,numer_of_users_in_partition)
#         for j in range(len(user_indexes)):
#             users[user_indexes[user_index]].preProcess_1(j,Z,hash_fuction, epsilon,l=None)
#             y[j] = users[user_indexes[user_index]].preProcess_1(j,Z,hash_fuction, epsilon,l=None)
#         y = np.array(y)
#         a = float(math.exp(epsilon)+1)/float(math.exp(epsilon)-1)*np.dot(y, Z)




def succinctHist(d, T, users,epsilon):
    n =len(users)
    private_bits = {}
    number_of_bits = int(math.log(d))
    global Z_1
    global Z_2

    partition = random_partition(n,number_of_bits)
    hash_function = create_random_Hash(100,users)
    Z_2 = createRandomMatrix(d,n)

    #each row is the bit representation of a heavy_hitter
    heavy_hitters = np.zeros([T,number_of_bits], dtype=int)

    for l in range(0,number_of_bits):
        bits_1 = np.zeros(n)
        bits_2 = np.zeros(n)

        for i in partition[l]:
            Z_1 = createRandomMatrix(2*T,n)
            (bit_1, bit_2) = users[i].return_private_bits( number_of_bits, hash_function, epsilon, l)
            bits_1[i] = bit_1
            bits_2[i] = bit_2
            #binary_value = int_to_binary(users[i].value,number_of_bits)

        bit_hash_value_estimate = float(np.exp(epsilon/2)+1) / float(np.exp(epsilon/2) - 1)*np.array(bits_1).dot(Z_1.transpose())
        for t in range(T):
            # decide on each of the bits with a majority vote
            zero_bit_count = bit_hash_value_estimate[binary_to_int(int_to_binary(t,number_of_bits)+'0')]
            one_bit_count =  bit_hash_value_estimate[binary_to_int(int_to_binary(t, number_of_bits) + '1')]
            heavy_hitters[t, l] = 0 if zero_bit_count > one_bit_count else 1
    print heavy_hitters
    # convert from binary representation int list of int values
    heavy_hitters_suspects = [ binary_to_int("".join(map(str,heavy_hitters[t,:])))  for t in range(T)]
    print heavy_hitters_suspects
    # estimate frequency for each of the suspected heavy hitters
    frequency ={}
    for value in heavy_hitters_suspects:
        frequency[value] = float(np.exp(epsilon/2)+1) / float(np.exp(epsilon/2) - 1)*bits_2.dot(Z_2[value,:])
    print frequency


class user_Node:
    def __init__(self, index , value):
        self.index = index
        self.value = int(value)


    def return_private_bits(self,number_of_bits, hash_fuction ,epsilon,partition_index):
        global Z_1
        global Z_2
        binary_rep = int_to_binary(self.value,number_of_bits)
        first_bit = Z_1[binary_to_int(int_to_binary(hash_fuction[self.value],number_of_bits)+binary_rep[partition_index]), self.index]
        first_bit = randomized_resopnse(first_bit, epsilon/2)
        second_bit = Z_2[self.value, self.index]
        second_bit = randomized_resopnse(second_bit, epsilon/2)
        return (first_bit, second_bit)



    


d = 500000
T=700
epsilon = 2.0
users = []
data = lines = [line.rstrip('\n') for line in open('test.txt')]
for i in range(len(data)):
    users.append(user_Node(i,data[i]))
succinctHist(d,T ,users,epsilon)


