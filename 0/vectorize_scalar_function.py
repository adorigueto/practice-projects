import numpy as np

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x <= y:
        result = x*y
    else: result = x/y

    return result

print(scalar_function(3,2))

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y
    """
    result = np.vectorize(scalar_function)
    return result(x,y)

x = np.array([3,2])
y = np.array([1,1])

print(vector_function(x,y))
