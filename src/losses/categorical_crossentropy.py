import numpy as np

from src.base import Loss
from src.models import eps


class SparseCategoricalCrossentropyLoss(Loss):

    @staticmethod
    def loss(y_hat, y):
        """
        Args:
            y_hat (numpy.ndarray): 2D prediction array with shape (m, k).
            y (list or tuple or numpy.ndarray): 2D array holding indices of correct classes with shape (m).

        Note:
            m: Number of examples.
            k: Number of classes.

        Returns:
            float: The mean corssentropy-loss over all examples.
        """
        m = len(y_hat)
        return -np.sum(np.log(np.clip(y_hat[(np.arange(m), y)], eps, 1.))) / m

    @staticmethod
    def gradient(y_hat, y):
        """
        Args:
            y_hat (numpy.ndarray): 2D prediction array with shape (m, k).
            y (list or tuple or numpy.ndarray): 2D array holding indices of correct classes with shape (m).

        Note:
            m: Number of examples.
            k: Number of classes.

        Returns:
            numpy.ndarray: Derivative of crossentropy-loss wrt y_hat for each example.  Returned array has shape (m, k).
        """
        m = len(y_hat)
        gradient = np.zeros_like(y_hat)
        gradient[(np.arange(m), y)] = -1 / np.clip(y_hat[(np.arange(m), y)], eps, 1.)
        return gradient


class CategoricalCrossentropyLoss(Loss):

    @staticmethod
    def loss(y_hat, y):
        """
        Args:
            y_hat (numpy.ndarray): 2D predicton array with shape (m, k).
            y (numpy.ndarray): 2D one-hot ground truth labels array with shape (m, k).

        Note:
            m: Number of examples.
            k: Number of classes.

        Returns:
            float: The mean crossentropy-loss over all examples.
        """
        m = len(y_hat)
        return -np.sum(y * np.log(np.clip(y_hat, eps, 1.))) / m

    @staticmethod
    def gradient(y_hat, y):
        """
        Args:
            y_hat (numpy.ndarray): 2D one-hot predicton array with shape (m, k).
            y (numpy.ndarray): 2D one-hot ground truth labels array with shape (m, k).

        Note:
            m: Number of examples.
            k: Number of classes.

        Returns:
            numpy.ndarray: Derivative of crossentropy-loss wrt y_hat for each example.  Returned array has shape (m, k).
        """
        return -y / np.clip(y_hat, eps, 1.)
