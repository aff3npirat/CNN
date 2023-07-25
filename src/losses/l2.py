import numpy as np

from src.base import Loss


class L2Loss(Loss):

    @staticmethod
    def loss(y_hat, y):
        m = len(y_hat)
        return (1/(2*m)) * np.sum(np.power(np.subtract(y, y_hat), 2))

    @staticmethod
    def gradient(y_hat, y):
        return np.subtract(y_hat, y)
