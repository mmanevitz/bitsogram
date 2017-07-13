import hashlib
import bitarray
def getSHA256HashArray(hashId, dataString):
    message = hashlib.sha256()

    message.update(str(hashId) + dataString)
    messageInBytes = message.digest()

    messageInBitArray = bitarray(endian='little')
    messageInBitArray.frombytes(messageInBytes)

    return messageInBitArray