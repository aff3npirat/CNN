import numpy as np

from src.nodes.node import Node


class Linear(Node):

    def __init__(self, x, w, b):
        """
        Args:
             x (Node): Input, x.output should be 2D np.ndarray with shape (m, x).
             w (Node): Weights, w.output shoud be 2D np.ndarray with shape (x, y).
             b (Node): Bias, b.output should be 1D np.ndarray with shape (y,).
        """
        super().__init__([x, w, b])

    def compute(self):
        """
        Computes x @ w + b.
        """
        x = self.input_nodes[0].output
        w = self.input_nodes[1].output
        b = self.input_nodes[2].output

        self.output = np.dot(x, w) + b

    def backpass(self):
        """
        Computes gradient of graph-output (L) wrt x, w, b.

        Notes:
            For inputs w, b the mean gradient over all features is calculated.   All resulting gradients are of type
            float.
        """
        self.gradients = {n: np.zeros_like(n.output, dtype=np.float) for n in self.input_nodes}

        x = self.input_nodes[0].output
        m = len(x)
        w = self.input_nodes[1].output

        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            self.gradients[self.input_nodes[0]] += np.dot(grad_cost, w.T)
            self.gradients[self.input_nodes[1]] += np.dot(x.T, grad_cost) / m
            self.gradients[self.input_nodes[2]] += np.sum(grad_cost, axis=0) / m
