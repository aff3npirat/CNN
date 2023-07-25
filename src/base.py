import abc

from src.nodes import placeholders


class Layer(abc.ABC):
    """
    Base class for all layersu used in model.

    A layer needs to be initialized before it can be used.  A layer has nodes that compute the layer's output.  If the
    layer has trainable parameters, their values can be accessed by calling get_param_values(self).  By calling
    output_shape the layers output shape can be accessed.  Input can be fed into layer by setting input_node.output
    and calling compute for all nodes (in respective order).

    Attributes:
        activation (string): Name of activation function used, needs to be key of ACTIVATION_REGISTRY.  Is None if
            layer has no activation function.
        name (string): Name of layer.  Does not affect layer behaviour.
        input_shape (list/tuple[int]): Containing shape of expected input of layer.
        input_node (Node): Input node for layer's graph.  For example output node of previous layer.
        output_node (Node): Node which is final node in layer's graph.
        parameters (list/tuple[Variable]): Contains all trainable parameters of this layer.
    """

    def __init__(self, input_shape=None):
        """
        Args:
            input_shape (list/tuple[int]): Optional; Expected shape of input.
        """
        self.activation = None
        self.name = None
        self.input_shape = input_shape
        self.input_node = placeholders.Variable()
        self.output_node = None
        self.parameters = []

    def output_shape(self):
        """
        Returns shape of output.
        """
        raise NotImplementedError("Should be implemented by subclass!")

    def initialize(self):
        """
        Initializes layer's parameters and creates nodes that compute layer's output.

        Note:
            Before calling this method, input_shape should be assigned.
        """
        raise NotImplementedError("Should be implemented by subclass!")

    def _build_graph(self):
        """
        Creates nodes for computational graph.
        """
        raise NotImplementedError("Should be implemented by subclass!")


class Loss:

    @staticmethod
    def loss(y_hat, y):
        pass

    @staticmethod
    def gradient(y_hat, y):
        pass


class Optimizer:
    """
    Base class for optimizer.

    Attributes:
        learning_rate (float)
    """

    def __init__(self, learning_rate):
        """
        Args:
            learning_rate (float)
        """
        self.learning_rate = learning_rate

    def update(self, trainables):
        raise NotImplementedError("Should be implemented by subclass!")

