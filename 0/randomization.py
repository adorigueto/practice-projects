import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    A = np.random.random([n,1])
    return A

n = 2
A = randomization(n)
print(A)