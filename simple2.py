import math
import numpy as np
import itertools
import random
from Crypto.Cipher import AES
from Crypto import Random
from user_side import user_Node
import bitstring
from tables import *
from config import config
from collections import Counter, defaultdict
import matplotlib.pyplot as plt






def int_to_binary(number, length):
    binary_string =format(number, '0' + str(length) + 'b')
    return np.array(map(int, binary_string))



def binary_to_int(value):
    return int(value, 2)


# devides the users into numberOfsubsets disjoint groups
def random_partition(users, numberOfsubsets):
    partition = {i: [] for i in range(len(users))}
    for user in users:
        partition[user.index] = random.randint(0, numberOfsubsets)
    print partition


def createRandomMatrix(T, n):
    return np.random.choice([-1, 1], size=(T, n))


def create_random_Hash(T, d):
    seed = random.randint(0, 1000000)
    return seed
    # hash_function = {}
    # for value in range(d):
    #     if value not in hash_function:
    #         hash_function[value] = random.randint(1, T - 2)
    # return hash_function


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


def hash_value(seed, value):
    random.seed(seed+value)
    return random.randint(1,T)



def randomized_resopnse(bit_x):
    prob = float(np.exp(config.epsilon)) / float(np.exp(config.epsilon) + 1)
    values = [bit_x, -bit_x]
    prob_array = [prob, 1-prob]
    # if the random float falls in the range [0, e^epsilon\e^epsilon)+1] return bit_x otherwise flip bit
    return np.random.choice([bit_x, -bit_x],p= [prob, 1-prob])


def frequency_oracle(value,user_values):
    frequency_estimate = 0
    for user_value in user_values:
        z = random.choice([1,-1])
        if value == user_value:
            frequency_estimate += config.cEpsilon*z*randomized_resopnse(z)
        else:
            frequency_estimate += config.cEpsilon*z
    return frequency_estimate







def random_partition(n, number_of_subsets):
    partition = {j: [] for j in range(number_of_subsets)}
    for i in range(n):
        set_number = random.randint(0, number_of_subsets - 1)
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

def oracle_test(data):
    real_count = dict(Counter(data))
    errors = []
    for value in range(10):
        estimate = frequency_oracle(value, data)
        error = real_count.get(str(value), 0) - estimate
        errors.append(error)
        print "value = %d real_count = %d ,estimate = %f, error =%f" % (
            value, real_count.get(str(value), 0), estimate, real_count.get(str(value), 0) - estimate)
        average_error = np.mean(errors)
    print "average error = %f, standard deviation = %f , max error = %f" % (
        np.mean(errors), np.std(errors), np.max(errors))
    errors = []
    for i in range(100):
        value = random.randint(10, config.d)
        estimate = frequency_oracle(value, data)
        error = real_count.get(str(value), 0) - estimate
        errors.append(error)
        print "value = %d, real_count = %d, estimate = %f, error =%f" % (
            value, real_count.get(str(value), 0), estimate, error)
    print "average error = %f, standard deviation = %f , max error = %f" % (
        np.mean(errors), np.std(errors), np.max(errors))





def invert_dict(d):
    d_inv = defaultdict(list)
    for k, v in d.items():
        d_inv[v].append(k)
    return d_inv

def hash_test(users,heavy_hitters):
    hash_domain_sizes= [100, 200, 300, 400, 500, 600, 700, 800 , 900, 1000 ,1500,2000, 3000]
    max_collisions = []
    for T in hash_domain_sizes:
        print "hash domain size = %d" %T
        hash_func = create_random_Hash(T, users)
        inv_hash = invert_dict(hash_func)
        for i in heavy_hitters:
            hash_value = hash_func[i]
            collisions = dict(Counter(hash_func.values()))
            total_number_collisions = len(inv_hash[hash_func[i]])
            intersecrion_with_h_h = list(set(inv_hash[hash_func[i]]) & set(heavy_hitters)-set([i]))
            print "value = %d, number of cllisions = %d, number_of_hh_collisions = %d"%(i,total_number_collisions, len(intersecrion_with_h_h))


        #print "T = %d ,max collisions = %f"%(T,np.max(collisions.values()))
        #print "T = %d ,max collisions = %f"%(T,np.max(collisions.values()))
        max_collisions.append(np.max(collisions.values()))

    plt.plot(hash_domain_sizes,max_collisions)
    plt.xlabel('hash_domain_size')
    plt.ylabel('number of collision')
    plt.show()

def explicitHist(value_range, values):
    frequency_dict = {}
    for value in value_range:
        frequency_dict[value] = frequency_oracle(value, values)
    return frequency_dict

def Hashtogram(d, T,epsilon, R, users, query_elements):
    n = len(users)
    a = np.zeros([T+1,R])
    Partition = random_partition(n, R)
    hashes = []
    count_by_random_value = [0 for i in range(0,T+1)]
    for partition_index in Partition:
        print partition_index
        hash_function = create_random_Hash(T, d)
        hashes.append(hash_function)
        for i in Partition[partition_index]:
            user = users[i]
            count_by_random_value[user.get_random_value_int(T)] += user.get_bit(hash_function)
        for t in range(1,T+1):
            for b in range(1,T+1):
                z = np.inner(int_to_binary(b ,T.bit_length()),int_to_binary(t,T.bit_length()))%2
                z = 1 if z == 0 else -1
                a[t,partition_index] += config.cEpsilon*z* count_by_random_value[b]

    #print "a[%d,%d] =%f" %(t,partition_index,a[t,partition_index])
    counts = {}
    for value in query_elements:
        vector =[a[hash_value(hashes[r],value) ,r]for r in range(R)]
        print vector
        counts[value] = R*np.median(vector)

    print counts
    return counts




def succinctHist(d, T,epsilon, users,user_values):
    n = len(users)
    private_bits = {}
    number_of_bits = int(math.log(config.d))
    partition = random_partition(n, number_of_bits)
    hash_function = create_random_Hash(config.T, users)
    #print hash_function

    hh_hash = [hash_function[value] for value in range(10)]
    hh_hash_0 = [(value,0) for value in hh_hash]
    hh_hash_1 = [(value,1) for value in hh_hash]
    # heavy_hitters = np.zeros([T, number_of_bits], dtype=int)
    heavy_hitters = np.zeros([len(hh_hash), number_of_bits], dtype=int)
    for l in range(number_of_bits):
        hash_bit_pairs = []
        print"********************"
        print "l=%d"%l
        print "********************"
        for i in partition[l]:
            value = user_values [i]

            binary_rep = int_to_binary(value, number_of_bits)
            hash_bit_pairs.append((hash_function[value],int(binary_rep[l])))

    #bit_hash_value_estimate = explicitHist(range(2*T),hash_bit_pairs)
        bit_hash_value_estimate = explicitHist(hh_hash_0+ hh_hash_1, hash_bit_pairs)
        print bit_hash_value_estimate
    # each row is the bit representation of a heavy_hitter
        #for t in range(T):
        for i in range(len(hh_hash)):
            t = hh_hash[i]
            # decide on each of the bits with a majority vote
            zero_bit_count = bit_hash_value_estimate[(t,0)]
            one_bit_count = bit_hash_value_estimate[(t,1)]
            heavy_hitters[i, l] = 0 if zero_bit_count > one_bit_count else 1
    print heavy_hitters

    # convert from binary representation int list of int values
    #heavy_hitters_suspects = [binary_to_int("".join(map(str, heavy_hitters[t, :]))) for t in range(T)]
    heavy_hitters_suspects = [binary_to_int("".join(map(str, heavy_hitters[t, :]))) for t in range(len(hh_hash))]
    #print heavy_hitters_suspects

    # estimate frequency for each of the suspected heavy hitters
    frequency = {}
    for value in heavy_hitters_suspects:
        # each row is the bit representation of a heavy_hitter
        frequency[value] = frequency_oracle(value, user_values)
    print frequency








class user_Node:
    def __init__(self, index, value):
        global T_bit_matrix
        self.index = index
        self.value = int(value)
        self.random_value_int = random.randint(1, T)
        self.random_value = T_bit_matrix[self.random_value_int]
        self.column = np.dot(T_bit_matrix, self.random_value)%2



    # def set_hash_function(self,hash_function):
    #     self.hash_function = hash_function

    def hash_value(self,seed, value):
        random.seed(seed+value)
        return random.randint(1,T+1)


        # hash_function = {}
        # for value in range(d):
        #     if value not in hash_function:
        #         hash_function[value] = random.randint(1, T - 2)
        # return hash_function

    def get_bit(self,hash_seed):
        self.hash_seed = hash_seed
        t = T_bit_matrix[hash_value(self.hash_seed ,self.value)]
        bit = np.inner(self.random_value,t)%2
        #translate from binary to +-1[
        bit = 1 if bit == 0 else -1
        return randomized_resopnse(bit)

    def get_random_value_int(self,T):
        return self.random_value_int
    def get_random_value(self):
        return self.random_value


def calculate_errors(real_count , result_count):
    errors = []
    for value in result_count:
        errors.append(abs(result_count[value]-real_count[value]))
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    return (avg_error, max_error, min_error)





def run_with_different_hash_domains(self,d, epsilon,R, users, real_count ,heavy_hitter_list):
    domain_sizes = [200*i for i in range(1,20)]
    all_avg_errors = []
    all_max_errors = []
    all_min_errors = []
    hh_avg_errors = []
    hh_max_errors = []
    hh_min_errors = []

    all_values = range(d)
    for T in domain_sizes:
        all_emperical_count = Hashtogram(d, T, config.epsilon, R, users,all_values)
        hh_emperical_count = Hashtogram(d, T, config.epsilon, R, users, heavy_hitter_list)
        (all_avg_error, all_max_error, all_min_error) = calculate_errors(real_count, all_emperical_count)
        (hh_avg_error, hh_max_error, hh_min_error) = calculate_errors(real_count, hh_emperical_count)
        all_avg_errors.append(all_avg_error)
        all_max_errors.append(all_max_error)
        all_min_errors.append(all_min_error)
        hh_avg_errors.append(hh_avg_error)
        hh_max_errors.append(hh_max_error)
        hh_min_errors.append(hh_min_error)

    plt.plot(domain_sizes,all_avg_error, label ="all_avg_error")
    plt.plot(domain_sizes,all_max_error, label ="all_max_error")
    plt.plot(domain_sizes,all_min_error, label = "all_min_error")
    plt.plot(domain_sizes,hh_avg_error , label = "hh_avg_error")
    plt.plot(domain_sizes,hh_max_error , label = "hh_max_error")
    plt.plot(domain_sizes,hh_min_error , label = "hh_min_error")
    plt.legend()
    plt.show











d = 500000
T = 700


config.reinitializeParameters()
users = []
data = lines = [int(line.rstrip('\n')) for line in open('test.txt')]
global T_bit_matrix
T_bit_matrix = np.array([int_to_binary(t, T.bit_length()) for t in range(0, T+1)])
for i in range(len(data)):
    users.append(user_Node(i, data[i]))


#oracle_test(data)
#explicitHist(xrange(d), data)
#hash_test(users,range(700))
#succinctHist(d, T, config.epsilon, users,data)
R =20
print Hashtogram(d, T,config.epsilon, R, users, range(10))



