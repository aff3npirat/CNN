import numpy as np

from src.nodes.node import Node


class Softmax(Node):
    def __init__(self, x):
        """
        Args:
            x (Node): Input, x.output should be 2D np.ndarray with shape (m, x).
        """
        super().__init__([x])

    def compute(self):
        """
        Computes the softmax function.

        Normalizes each row-vector of x, so that result is np.ndarray with sum equal to m.  Result is rounded to two
        decimal places.

        Notes:
            Range of output is [0, 1].
        """
        x = self.input_nodes[0].output
        x -= np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        denom = np.sum(exp, axis=1, keepdims=True)
        self.output = np.round(np.divide(exp, denom), 2)

    def backpass(self):
        """
        Computes gradient of graph-output (L) wrt x.
        """
        self.gradients = {n: np.zeros_like(n.output, dtype=np.float) for n in self.input_nodes}

        y = self.output
        y_ = y.reshape(*y.shape, 1)
        n = y.shape[1]
        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            gradient = np.identity(n) - y_
            gradient = np.multiply(gradient.transpose(0, 2, 1), y_)
            self.gradients[self.input_nodes[0]] += np.matmul(gradient, grad_cost.T)[:, :, 0]
