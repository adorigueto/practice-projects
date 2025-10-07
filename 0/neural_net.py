import numpy as np

def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    w = np.array(weights)
    x = np.array(inputs)

    result = np.tanh(np.sum(w * x))
    out = np.array([[result]])

    return out

w = [[0.96960507,0.53815731]]
x = [[0.4180937,0.32173488]]

print(neural_network(w,x).shape)
print(neural_network(w,x))
