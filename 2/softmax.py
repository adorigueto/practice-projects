import numpy as np

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    # Compute the dot products: theta_j · x_i for all j and i
    # Result will be (k, n) matrix where element [j,i] = theta_j · X[i]
    dot_products = theta @ X.T  # shape: (k, n)
    
    # Divide by temperature parameter
    scaled_dot_products = dot_products / temp_parameter  # shape: (k, n)
    
    # Numerical stability: subtract max from each column (for each data point)
    # Find max for each data point across all classes
    max_vals = np.max(scaled_dot_products, axis=0, keepdims=True)  # shape: (1, n)
    
    # Subtract max from each column to prevent overflow
    stable_exponents = scaled_dot_products - max_vals  # shape: (k, n)
    
    # Compute exponentials
    exp_vals = np.exp(stable_exponents)  # shape: (k, n)
    
    # Compute denominator (sum over classes for each data point)
    denominators = np.sum(exp_vals, axis=0, keepdims=True)  # shape: (1, n)
    
    # Compute probabilities
    H = exp_vals / denominators  # shape: (k, n)
    
    return H


#X = np.array([[3,3,3],[3,3,3],[2,2,2]])
#theta = np.zeros(X.shape[0])
#temp_parameter = 0.1

#print(compute_probabilities(X, theta, temp_parameter))

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    n = X.shape[0]  # number of data points
    k = theta.shape[0]  # number of classes
    
    # Get probabilities using the previous function
    probabilities = compute_probabilities(X, theta, temp_parameter)  # shape: (k, n)
    
    # Create indicator matrix [Y_i == j]: for each data point, 1 at true label, 0 elsewhere
    indicator = np.zeros((k, n))
    indicator[Y, np.arange(n)] = 1  # set 1 at positions (Y[i], i)
    
    # Compute cross-entropy loss
    # Only consider the probabilities of the true classes
    true_class_probs = probabilities[Y, np.arange(n)]
    log_probs = np.log(true_class_probs)
    cross_entropy = -np.sum(log_probs) / n
    
    # Compute regularization term (L2 norm of theta matrix)
    regularization = (lambda_factor / 2) * np.sum(theta ** 2)
    
    # Total cost
    total_cost = cross_entropy + regularization
    
    return total_cost


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    n = X.shape[0]  # number of data points
    k = theta.shape[0]  # number of classes
    
    # Step 1: Compute probabilities
    probabilities = compute_probabilities(X, theta, temp_parameter)  # shape: (k, n)
    
    # Step 2: Create an indicator matrix with 1s at true labels
    indicator = sparse.coo_matrix((np.ones(n), (Y, np.arange(n))), shape=(k, n)).toarray()
    
    # Step 3: Calculate the gradient main term
    # From the derivative equation: ∂J/∂θ_m = -1/(τn) * ∑[ x⁽ⁱ⁾( [y⁽ⁱ⁾==m] - p(y⁽ⁱ⁾=m|x⁽ⁱ⁾,θ) ) ] + λθ_m
    gradient = (-1 / (temp_parameter * n)) * (indicator - probabilities) @ X
    
    # Step 4: Add the regularization term
    gradient += lambda_factor * theta
    
    # Step 5: Update theta using gradient descent
    theta = theta - alpha * gradient
    
    return theta


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3
    
    return train_y_mod3, test_y_mod3

