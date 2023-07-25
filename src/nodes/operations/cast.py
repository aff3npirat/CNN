import abc
import numpy as np

from src.nodes.node import Node


class Cast(Node, abc.ABC):

    def __init__(self, a, dtype):
        """
        Args:
            a (Node):
            dtype (numpy.dtype): Data-type to cast to.

        Note:
            `a.output` should be numpy.ndarray.
        """
        super().__init__([a])
        self._dtype = dtype


class CastForwardOnly(Cast):
    """
    Casts on forward pass.
    """

    def compute(self):
        a = self.input_nodes[0].output
        self.output = a.astype(self._dtype)

    def backpass(self):
        self.gradients = {n: np.zeros_like(n.output) for n in self.input_nodes}

        for node in self.output_nodes:
            self.gradients[self.input_nodes[0]] += node.gradients[self]
