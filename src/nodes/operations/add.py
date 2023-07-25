import numpy as np

from src.nodes.node import Node


class Add(Node):

    def __init__(self, a, b):
        """
        Args:
            a (Node): First summand.
            b (Node): Second summand.

        Note:
            Expects `a.output` and `b.output` to have shape which can be broadcasted together.
        """
        super().__init__([a, b])

    def compute(self):
        a = self.input_nodes[0].output
        b = self.input_nodes[1].output
        self.output = a + b

    def backpass(self):
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        for node in self.output_nodes:
            grad_cost = node.gradients[self]

            self.gradients[self.input_nodes[0]] += grad_cost
            self.gradients[self.input_nodes[1]] += np.sum(grad_cost, axis=0)
