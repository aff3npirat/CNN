import numpy as np

from src.base import Layer
from src.nodes.operations import flatten


class Flatten(Layer):
    """
    Class representing a Flatten layer.

    A Flatten layer, flattens the input to 1D.
    """

    def __init__(self, **kwargs):
        """
        Keyword Args:
            See base class.
        """
        super().__init__(**kwargs)

    def output_shape(self):
        """
        See base class Layer.
        """
        return (np.prod(self.input_shape),)

    def initialize(self):
        """
        See base class.
        """
        self._build_graph()

    def _build_graph(self):
        """
        See base class.
        """
        self.output_node = flatten.Flatten(self.input_node)
