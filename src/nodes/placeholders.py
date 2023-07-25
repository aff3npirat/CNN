import numpy as np

from src.nodes.node import Node


class Variable(Node):
    """
    Node that only provides a value for graph.

    On backpass a Variable stores gradient of graph-output wrt its output.
    """

    def __init__(self, init_value=None):
        super().__init__()
        self.output = init_value

    def compute(self):
        pass

    def backpass(self):
        self.gradients[self] = np.zeros_like(self.output)

        for node in self.output_nodes:
            self.gradients[self] += node.gradients[self]


class GradPlaceholder(Node):
    """
    Node that propagates a given gradient to input nodes.

    Can be used for providing an initial gradient in graph.
    """

    def __init__(self, a):
        super().__init__([a])

    def compute(self):
        self.output = self.input_nodes[0].output

    def backpass(self, grad_cost=None):
        self.gradients = {self.input_nodes[0]: grad_cost}
