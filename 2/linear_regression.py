import numpy as np


def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    
    I = np.identity(X.shape[1])
    A = X.T @ X + lambda_factor * I
    b = X.T @ Y

    theta = np.linalg.solve(A, b)
    
    return theta


X_data = [[0.4250526, 0.22127994, 0.53211637]
 ,[0.89807472, 0.36201998, 0.85817951]
 ,[0.05806973, 0.44981903, 0.75307626]
 ,[0.56849054, 0.69020339, 0.94912018]
 ,[0.41915046, 0.92679984, 0.88282992]
 ,[0.45204143, 0.20688909, 0.20610468]
 ,[0.71173957, 0.40821322, 0.65349384]
 ,[0.9585099, 0.98398955, 0.72725041]
 ,[0.79945342, 0.52970394, 0.31966112]
 ,[0.00715593, 0.58907755, 0.63007761]
 ,[0.83269188, 0.44610012, 0.09422533]
 ,[0.13461568, 0.99005527, 0.64619322]
 ,[0.94341996, 0.8386825, 0.00118375]]
Y_data = [0.92050857, 0.57135775, 0.67204214, 0.75239476, 0.89930585, 0.01749785,
 0.91615574, 0.65216976, 0.31795273, 0.61899572, 0.59498881, 0.50015733,
 0.68438415]

X = np.array(X_data)
Y = np.array(Y_data)

lambda_factor = 0.24274550753327584

print(closed_form(X, Y, lambda_factor))

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
