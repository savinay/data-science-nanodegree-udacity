import numpy as np
import math
# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    res = []
    sum_exp = 0
    for num in L:
        sum_exp += math.exp(num)
    for num in L:
        res.append(math.exp(num)/sum_exp)
    return res	


#---------------Alternate Solution-----------------

# import numpy as np

# def softmax(L):
#     expL = np.exp(L)
#     sumExpL = sum(expL)
#     result = []
#     for i in expL:
#         result.append(i*1.0/sumExpL)
#     return result
