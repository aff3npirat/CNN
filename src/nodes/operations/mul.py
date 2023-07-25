import numpy as np

from src.nodes.node import Node


class MatMul(Node):
    """
    Node computing matrix multiplication.
    """

    def __init__(self, a, b):
        """
        Args:
            a (Node): First operand.
            b (Node): Second operand.

        Note:
            `a.output` and `b.output` should be `numpy.ndarray`.
        """
        super().__init__([a, b])

    def compute(self):
        a = self.input_nodes[0].output
        b = self.input_nodes[1].output
        self.output = np.dot(a, b)

    def backpass(self):
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        a = self.input_nodes[0].output
        b = self.input_nodes[1].output
        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            if np.prod(grad_cost.shape) == 1:
                grad_cost = grad_cost.item()

            self.gradients[self.input_nodes[0]] += np.dot(grad_cost, b.T)
            self.gradients[self.input_nodes[1]] += np.dot(a.T, grad_cost)
