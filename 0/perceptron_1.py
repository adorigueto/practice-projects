import numpy as np

def perceptron(x, y, theta = 0, theta_not = 0, T=5):
    i = 1

    for key, value_list in x.items():
            x[key] = np.array(value_list)

    for T in range(0,T):
        for i in range(len(x)-1):
            if np.linalg.norm(y[i]*(theta*x[i]+theta_not)) <= 0:
                theta = theta + y[i]*x[i]
                theta_not = theta_not + y[i]
    return theta, theta_not


theta = 0
theta_not = 0
x = {0:[-4, 2], 1:[-2, 1], 2:[-1, -1], 3:[2, 2], 4:[1, -2]}
y = [1,1,-1,-1,-1]
T = 6

def n(x):
    for key, value_list in x.items():
            x[key] = np.array(value_list)
    return x

x = n(x)
print(x[0])

print(y[0])

print(np.linalg.norm([1]*(theta*x[1]+theta_not)))

print(perceptron(x,y,theta,theta_not, T))