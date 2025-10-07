import numpy as np


feature_vector = np.array([1,1])

matrix_data = [[1, 1],
               [1, 1],
               [1, 1]]
feature_matrix = np.array(matrix_data)

labels = np.array([1,1,-1])

label = 1
theta = np.array([1,1])
theta_0 = 1

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    
    z = label*(theta@feature_vector + theta_0)

    if z >= 1:
        loss = 0
    else:
        loss = 1-z
    
    return z, loss

print(hinge_loss_single(feature_vector, label, theta, theta_0))

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """

    loss = np.empty((len(labels)))

    for i in range(len(labels)):
        z = labels[i]*(theta@feature_matrix[i] + theta_0)
        if z >= 1:
            loss[i] = 0
        else:
            loss[i] = 1-z
    
    return np.mean(loss)

print(hinge_loss_full(feature_matrix, labels, theta, theta_0))