import abc

from src.base import Layer
from src.nodes.operations import pooling


class Pool(Layer, abc.ABC):
    """
    Base class for pooling layers.

    Attributes:
        filter_size (list/tuple[int, int]): Size of filter.
        stride (list/tuple[int, int]): Step size filter is moved.
    """

    def __init__(self, filter_size=(2, 2), stride=(2, 2), **kwargs):
        """
        Args:
            filter_size (list/tuple[int, int]): Optional; Size of filter, should be [X_size, Y_size].
            stride (list/tuple[int, int]): Optional; Step size filter is moved, should be [X_step, Y_step].

        Keyword Args:
            See base class.
        """
        super().__init__(**kwargs)
        self.filter_size = filter_size
        self.stride = stride

    def output_shape(self):
        """
        See base class.
        """
        out_shape = [(i - f)//s + 1 for i, f, s in zip(self.input_shape, self.filter_size, self.stride)]
        return (*out_shape, self.input_shape[-1])


class MaxPool(Pool):
    """
    Class representing a MaxPooling layer.

    Extends class Pool by _build_graph(self) method.
    """

    def __init__(self, filter_size=(2, 2), stride=(2, 2), **kwargs):
        """
        Args:
            See base class.
        """
        super().__init__(filter_size, stride, **kwargs)

    def initialize(self):
        """
        See base class.
        """
        self._build_graph()

    def _build_graph(self):
        """
        See base class.
        """
        self.output_node = pooling.MaxPool(self.input_node, self.filter_size, self.stride)


class MeanPool(Pool):
    """
    Class representing a MeanPooling layer.
    """

    def __init__(self, filter_size=(2, 2), stride=(2, 2), **kwargs):
        """
        Args:
            See base class.
        """
        super().__init__(filter_size, stride, **kwargs)

    def initialize(self):
        """
        See base class.
        """
        self._build_graph()

    def _build_graph(self):
        """
        See base class.
        """
        self.output_node = pooling.MeanPool(self.input_node, self.filter_size, self.stride)
