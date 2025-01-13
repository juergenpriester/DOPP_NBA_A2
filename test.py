

from functools import reduce
import numpy as np


def factors(n):
    return set(reduce(
        list.__add__,
        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


n = 10

factors_list = np.array(list(factors(8)))
print(factors_list)

max_cols = 4
factors_list = factors_list[1 < factors_list <= max_cols]
print(factors_list)
