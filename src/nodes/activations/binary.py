import numpy as np

from src.nodes.node import Node


class Binary(Node):

    def __init__(self, x):
        super().__init__([x])

    def compute(self):
        x = self.input_nodes[0].output
        out = np.zeros_like(x)
        out[x >= 0] = 1
        self.output = out

    def backpass(self):
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            self.gradients[self.input_nodes[0]] += grad_cost
