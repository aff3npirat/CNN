import numpy as np

from losses import loss


class L1Loss(loss.Loss):

    @staticmethod
    def loss(y_hat, y):
        m = len(y_hat)
        n = len(y_hat[0])
        return np.sum(np.abs(np.subtract(y, y_hat))) / (m * n)

    @staticmethod
    def gradient(y_hat, y):
        m = len(y_hat)
        n = len(y_hat[0])
        gradient = np.zeros_like(y_hat)
        for k in range(m):
            for i in range(n):
                if y[k, i] < y_hat[k, i]:
                    gradient[k, i] = -1
                elif y_hat[k, i] < y[k, i]:
                    gradient[k, i] = 1
        return gradient / n
