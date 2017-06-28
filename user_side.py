import numpy as np
import random
import Crypto
from Crypto.Cipher import AES
from bitarray import bitarray
from bitstring import BitArray
from bitstring import BitArray

def preProcess(Z,partition,hash_fuction, encoding):
    print "hello"

def randomized_resopnse(bit_x, epsilon):
    threshold = float(np.exp(epsilon)) / float(np.exp(epsilon) + 1)
    # if the random float falls in the range [0, e^epsilon\e^epsilon)+1] return bit_x otherwise flip bit
    if random.random() <= threshold:
        return bit_x
    else:
        return -bit_x

def int_to_binary(value,length):
    return format(value, '0'+str(length)+'b')

def binary_to_int(value):
    return int(value, 2)


class user_Node:
    def __init__(self, index , value):
        self.index = index
        self.value = value


    def return_private_bits(self,Z_1,Z_2,number_of_bits, hash_fuction ,epsilon,partition_index):

        binary_rep = int_to_binary(self.value,number_of_bits)
        first_bit = Z_1[binary_to_int(int_to_binary(hash_fuction[self.value],number_of_bits)+binary_rep[partition_index]), self.index]
        first_bit = randomized_resopnse(first_bit, epsilon/2)

        second_bit = Z_2[self.value, self.index]
        second_bit = randomized_resopnse(second_bit, epsilon/2)
        return (first_bit, second_bit)





