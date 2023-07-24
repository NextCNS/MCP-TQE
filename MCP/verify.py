# constraint violate
import numpy as np


def verify_constraint_1(x, k):
    if np.sum(x, axis=0) > k:
        #print("constraint 1 is violated")
        return False
    return True


def verify_constraint_2(x, y, A):
    for i in range(0, len(y)):
        cover = 0
        for j in range(0, len(x)):
            cover += x[j] * A[i, j]

        if cover < y[i]:
            #print("constraint 2 is violated")
            return False

    return True


def verify(x, y, k, A, sol):
    condition1 = (sum(x) < k + 0.5)
    condition2 = verify_constraint_2(x, y, A)
    return condition1, condition2
