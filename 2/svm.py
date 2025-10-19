import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    clf = LinearSVC(random_state=0, C=0.1)
    model = clf.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)

    return pred_test_y


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    clf = LinearSVC(random_state=0, C=0.1)
    model = clf.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)

    return pred_test_y


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)


X_train_data = [[0.44522087, 0.01852058, 0.39037619]
 ,[0.56847382, 0.98794971, 0.68717895]
 ,[0.97830055, 0.94748604, 0.66611041]
 ,[0.49555671, 0.42664438, 0.73287565]
 ,[0.21786142, 0.46449987, 0.47614957]
 ,[0.84735872, 0.23182911, 0.29040638]
 ,[0.11949098, 0.91335453, 0.8233649]
 ,[0.37658498, 0.63161317, 0.61553418]
 ,[0.0271821, 0.1995346, 0.49752224]
 ,[0.42669455, 0.34311284, 0.81992418]
 ,[0.88987399, 0.34308672, 0.82166006]
 ,[0.09650248, 0.8972288, 0.48686172]
 ,[0.7481516, 0.41348867, 0.23808242]]
y_train_data = [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
X_test_data = [[0.59029883, 0.38602541, 0.89682694]
 ,[0.48473453, 0.69363614, 0.89132118]
 ,[0.91490493, 0.37957456, 0.34085703]
 ,[0.4967469, 0.72542138, 0.64396477]
 ,[0.16690352, 0.67500152, 0.97794473]
 ,[0.20547517, 0.11884393, 0.54248464]
 ,[0.20493198, 0.81932967, 0.36702977]
 ,[0.90409479, 0.62148209, 0.64725414]
 ,[0.53609727, 0.81980708, 0.10542827]
 ,[0.76058487, 0.47014701, 0.57073069]
 ,[0.82196735, 0.09283943, 0.71887154]
 ,[0.20313452, 0.12574436, 0.65604909]
 ,[0.21431675, 0.68899456, 0.60160062]]
 
train_x = np.array(X_train_data)
train_y = np.array(y_train_data)
test_x = np.array(X_test_data)

print(one_vs_rest_svm(train_x, train_y, test_x))