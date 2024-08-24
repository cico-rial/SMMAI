"""This python module is for storing all 
the multiple-used function inside my homework3"""

import numpy as np
import matplotlib.pyplot as plt

def train_test_split(X, Y, N_train):
    """
    Divide the input dataset in training and testing part.

    train_test_split(X, Y, N_train):
        X : (d,N) shaped numpy dataset
        y : (N,) shaped numpy dataset
        N_train : number of data point for the training

    return X_train, Y_train, X_test, Y_test
    """
    try:
        d, N = X.shape
    except:
        N = X.shape[-1]

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


def backtracking(f, grad_f, x, multivariate):
    """
    This function is a simple implementation of the backtracking algorithm for
    the GD (Gradient Descent) method.

    f: function. The function that we want to optimize.
    grad_f: function. The gradient of f(x).
    x: ndarray. The actual iterate x_k.
    multivariate: boolean. Indicate whether the function f
    is multivariate or not.
    """

    if multivariate:
        norm = np.linalg.norm
    else:
        norm = np.abs

    alpha = 1
    c = 0.8
    tau = 0.25

    while f(x - alpha * grad_f(x, X, Y), X, Y) > f(x, X, Y) - c * alpha * norm(grad_f(x, X, Y)) ** 2:
        alpha = tau * alpha

        if alpha < 1e-3:
            break
    return alpha


def GD(f, grad_f, x0, kmax, tolf, tolx, back_tracking=False, alpha=1, multivariate=True):
    """
    This function implements the Gradient Descent algorithm.
    """
    if multivariate:
        norm = np.linalg.norm
    else:
        norm = np.abs

    # Initialize x_k
    x_k = x0

    # counter
    k = 0

#     # Initialize the outputs
#     x = [x0]
#     f_val = [f(x0)]
#     grads = [grad_f(x0)]
#     err = [norm(grad_f(x0))]

    # Loop
    condition = True
    while condition:
        # Update alpha
        if back_tracking:
            alpha = backtracking(f, grad_f, x_k, multivariate)
        # Update x
        x_k = x_k - alpha * grad_f(x_k, X, Y)

#         # Update outputs
#         x.append(x_k)
#         f_val.append(f(x_k))
#         grads.append(grad_f(x_k))
#         err.append(norm(grad_f(x_k)))

        # check criteria
        condition1 = norm(grad_f(x_k, X, Y)) > (tolf * norm(grad_f(x0, X, Y)))
        condition2 = norm(x_k) > tolx
        condition3 = k < kmax
        condition = condition1 and condition2 and condition3

        # update k
        k += 1

    return x_k#, k, np.array(x), np.array(f_val), np.array(grads), np.array(err)
