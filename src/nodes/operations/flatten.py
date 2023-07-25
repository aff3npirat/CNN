import numpy as np

from src.nodes.node import Node


class Flatten(Node):
    """
    Node that flattens its input.
    """

    def __init__(self, x):
        """
        Args:
            x (Node): Input to be flattend, x.output should have shape MxX (where X can be any shape).
        """
        super().__init__([x])

    def compute(self):
        """
        Flattens x.output, so that it becomes a MxX_flat array.
        """
        x = self.input_nodes[0].output
        m, h_in, w_in, c = x.shape
        self.output = np.reshape(x, (m, h_in * w_in * c))

    def backpass(self):
        """
        Reshapes upstream gradient to x.output's shape.
        """
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        d_in = np.shape(self.input_nodes[0].output)
        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            self.gradients[self.input_nodes[0]] += grad_cost.reshape(d_in)
