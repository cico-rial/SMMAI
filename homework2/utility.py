"""
This python module is for storing all
the multiple-used function inside my homework2
"""

import numpy as np

def train_test_split(X, Y, N_train):
    """
    Divide the input dataset in training and testing part.

    train_test_split(X, Y, N_train):
        X : (d,N) shaped numpy dataset
        y : (N,) shaped numpy dataset
        N_train : number of data point for the training

    return X_train, Y_train, X_test, Y_test
    """
    d, N = X.shape

    # Define the array of indices
    idx = np.arange(0, N)

    # Shuffle the indices
    np.random.shuffle(idx)

    # Extract train and test indices
    train_idx = idx[:N_train]
    test_idx = idx[N_train:]

    # Extract data
    X_train = X[:, train_idx]
    Y_train = Y[:, train_idx]

    X_test = X[:, test_idx]
    Y_test = Y[:, test_idx]

    return X_train, Y_train, X_test, Y_test


def centroid(X, keepdims=True):
    """
    Return the centroid of the dataset.

    centroid(X, keepdims=True):
        X : input dataset
    """
    return np.mean(X, axis=1, keepdims=keepdims)