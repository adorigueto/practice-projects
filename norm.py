import numpy as np

def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """

    s = np.linalg.norm(A+B)

    return s


A = np.array([3,2,2])
B = np.array([1,-2,5])

print(norm(A,B))
