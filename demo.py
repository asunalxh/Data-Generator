import itertools
import json
from copy import copy
import numpy as np

x = np.array([
    [1,2,3,4],
    [1,3,5,6],
    [2,3,4,1],
    [2,1,4,5],
    [3,5,5,5]
])

selector = np.full(x.shape[0],True)
selector = selector & (x[:,0] > 1)
selector &= x[:,1] < 2
print(selector)
print(x[selector])