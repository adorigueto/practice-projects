import numpy as np
import random


#%% UTILS FUNCTION


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


#%% PERCEPTRON FULL CYCLE FUNCTION


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """

    updated_theta = np.array(len(current_theta.shape))
    updated_theta_0 = 0.0

    if np.linalg.norm(label*(current_theta*feature_vector+current_theta_0)) <= 0:
        updated_theta = current_theta + label*feature_vector
        updated_theta_0 = current_theta_0 + label

    result = (updated_theta, updated_theta_0)

    return updated_theta, updated_theta_0


# Test samples
feature_vector = np.array([-0.07862224,-0.30837509,-0.0280018,0.01853348,0.43271934,-0.40370374,-0.07420045,-0.07714774,0.09458982,0.35117407])
label = 1
current_theta = np.array([0.2971947,-0.12213542,-0.18381982,0.31047926,-0.38462748,-0.34077605,0.17526027,-0.28490908,0.21563437,-0.30283126])
current_theta_0 = 0.0806381974533076

# Result
print(perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0))


#%% PERCEPTRON FULL CYCLE FUNCTION


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
    
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0
    nsamples, nfeatures = feature_matrix.shape
    
    for t in range(T):
        for i in get_order(nsamples):
            updated_theta, updated_theta_0 = perceptron_single_step_update(feature_vector=feature_matrix[i], label=labels[i],
                                          current_theta=current_theta, current_theta_0=current_theta_0)
            current_theta = updated_theta
            current_theta_0 = updated_theta_0

    return updated_theta, updated_theta_0


# Test sample
matrix_data = [[1, 1],
               [1, 1],
               [1, 1]]
feature_matrix = np.array(matrix_data)

labels = np.array([1,1,-1])

T = 5

# Result
result = perceptron(feature_matrix, labels, T)
print(result[0])