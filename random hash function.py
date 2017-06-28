from Crypto.Cipher import AES
from Crypto import Random
from bitarray import bitarray
from bitstring import BitArray
from bitstring import BitArray

def hex_string_to_int_array(self, hex_string):
    result = []
    for i in range(0, len(hex_string), 2):
        result.append(int(hex_string[i:i + 2], 16))
    return result

