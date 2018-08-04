import numpy as np
import math
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    res = 0
    for i in range(len(Y)):
        res += Y[i] * math.log(P[i]) + (1 - Y[i]) * math.log(1 - P[i])
    return -res


###############  Alternate Solution  ##################

# import numpy as np

# def cross_entropy(Y, P):
#     Y = np.float_(Y)
#     P = np.float_(P)
#     return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))