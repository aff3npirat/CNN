import numpy as np

from src.base import Optimizer


class Momentum(Optimizer):
    """
    A simple momentum optimizer.
    """

    def __init__(self, learning_rate, mu=0.9):
        """
        Args:
            learning_rate (float):
            mu (float):
        """
        super().__init__(learning_rate)
        self.mu = mu
        self._v = {}

    def update(self, trainables):
        for node in trainables:
            if node not in self._v.keys():
                self._v[node] = np.zeros_like(node.output)

            v = self.mu * self._v[node] - self.learning_rate * node.gradients[node]
            node.output += v
            self._v[node] = v
