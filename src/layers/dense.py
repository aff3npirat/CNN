import math
import numpy as np

from src.base import Layer
from src.layers import ACTIVATION_REGISTRY
from src.nodes import placeholders
from src.nodes.operations import linear


class Dense(Layer):
    """
    Class representing a fully connected layer.

    Attributes:
        units (int): Number of nodes.
        weights (numpy.array): Stores weights and bias for this layer.  Has shape (input_shape+1)x(units).
    """

    def __init__(self, units, activation='relu', **kwargs):
        """
        Args:
            units: Number of units.
            activation: See base class.

        Keyword Args:
            See base class.
        """
        super().__init__(**kwargs)
        self.activation = activation
        self.units = units

    def output_shape(self):
        """
        See base class.
        """
        return (self.units,)

    def initialize(self):
        """
        See base class.
        """
        limit = math.sqrt(6 / (self.input_shape[0] + self.units))
        weights = np.array(np.random.uniform(-limit, limit, (self.input_shape[0], *self.output_shape())), dtype=np.float64)
        bias = np.zeros(self.output_shape(), dtype=np.float64)
        self.parameters = [placeholders.Variable(weights), placeholders.Variable(bias)]
        self._build_graph()

    def _build_graph(self):
        """
        See base class.
        """
        lin = linear.Linear(self.input_node, self.parameters[0], self.parameters[1])
        act = ACTIVATION_REGISTRY[self.activation](lin)
        self.output_node = act