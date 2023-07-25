import numpy as np

from src.nodes.node import Node


class Relu(Node):
    """
    Node that applies the ReLU function to its input.
    """

    def __init__(self, x):
        """
        Args:
            x (Node): Input to which relu is applied, x.output should be array_like.
        """
        super().__init__([x])

    def compute(self):
        """
        Applies relu function to input.
        """
        x = self.input_nodes[0].output
        self.output = np.maximum(x, 0)

    def backpass(self):
        """
        Computes gradient of graph-output (L) wrt x.

        Since y_i= max(x_i, 0) for x_i in x, dy_i/dx_i= 1 if x_i > 0 else 0 and dL/dx_i eqauls upstream gradient
        multiplied by dy/dx.
        """
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            gradient = np.array(self.output)
            gradient[gradient > 0] = 1

            self.gradients[self.input_nodes[0]] += np.multiply(gradient, grad_cost)
