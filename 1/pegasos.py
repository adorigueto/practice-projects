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
    

#%% PEGASOS SINGLE STEP

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    if label * (np.dot(theta, feature_vector) + theta_0) <= 1:
        # For theta: apply regularization and add the feature vector component
        updated_theta = (1-eta*L) * theta + eta * label * feature_vector
        # For theta_0: no regularization, just add the label component
        updated_theta_0 = theta_0 + (eta * label)
    else:
        # Only apply regularization to theta, no change to theta_0
        updated_theta = (1 - eta * L) * theta
        updated_theta_0 = theta_0

    return updated_theta, updated_theta_0


# Test samples
feature_vector = np.array([-0.07862224,-0.30837509,-0.0280018,0.01853348,0.43271934,-0.40370374,-0.07420045,-0.07714774,0.09458982,0.35117407])
label = -1
L = 0.3002512691409366
eta = 0.663113495582509
theta = np.zeros(feature_vector.shape[0])
theta_0 = 0.0

# Result
print("theta: ", pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0)[0],
      "\ntheta_0: ", pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0)[1])


#%% PEGASOS


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    nsamples, nfeatures = feature_matrix.shape
    theta = np.zeros(nfeatures)
    theta_0 = 0.0
    
    # Initialize update counter
    t = 1  # Update counter starts at 1
    
    for iteration in range(T):
        # Get the ordering for this iteration
        order = get_order(nsamples)
        
        for i in order:
            # Calculate learning rate for this update
            eta = 1.0 / np.sqrt(t)
            
            # Perform single step update
            theta, theta_0 = pegasos_single_step_update(
                feature_matrix[i], 
                labels[i], 
                L, 
                eta, 
                theta, 
                theta_0
            )
            
            # Increment update counter
            t += 1
    
    return theta, theta_0
