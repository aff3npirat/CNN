import numpy as np


def softmax_accuracy(y_hat, y):
    """
    Calculates how often predictions matches one-hot labels.

    Args:
        y_hat (numpy.ndarray): Output of softmax layer with shape (M, X).
        y (list or tuple or numpy.ndarray): Target classes as indices with shape (M).

    Returns:
        float: Percentage of matches.
    """
    return (np.argmax(y_hat, axis=1) == y).mean()


def accuracy(y_hat, y):
    """
    Calculates how often predictions equal labels.

    Args:
        y_hat (numpy.ndarray): Predictions.
        y (numpy.ndarray): Ground truth values with same shape as y_hat.

    Returns:
        float: Percentage of matches.
    """
    return (y_hat == y).mean()
