import math
import numpy as np

from src.base import Layer
from src.layers import ACTIVATION_REGISTRY
from src.nodes import placeholders
from src.nodes.operations import convolution, fast_convolution


class Conv(Layer):
    """
    Class representing a convolutional layer.

    Attributes:
        num_kernels (int): Number of filters used for each convolution operation.  Determines output channels.
        d_kernel (list/tuple[int, int]): Size of filters.
        padding (bool): If True zero padding will be applied to input.
    """

    def __init__(self, num_kernels, d_kernel, activation='relu', padding=False, **kwargs):
        """Initializes convolutional layer with num_kernels, kernel_dim, stride and padding.

        Args:
            num_kernels (int): Number of kernels.
            d_kernel (list/tuple[int, int]): Size of filters, so that d_kernel= [X_size, Y_size].
            activation (string): Optional; See base class.
            padding (bool): Optional; If True zero padding will be applied.

        Keyword Args:
            See base class.

        Raises:
            ValueError: Only when d_kernel contains at least one even number and padding is true.
        """
        super().__init__(**kwargs)
        if padding is True:
            if d_kernel[0] % 2 == 0 or d_kernel[1] % 2 == 0:
                raise ValueError("Kernel size should be odd when padding is enabled!")

        self.activation = activation
        self.num_kernels = num_kernels
        self.d_kernel = d_kernel
        self.padding = padding

    def output_shape(self):
        """
        See base class.
        """
        if self.padding is True:
            out_shape = self.input_shape[:-1]
        else:  # if self.padding is False
            out_shape = (i - k + 1 for i, k in zip(self.input_shape[:-1], self.d_kernel))
        return (*out_shape, self.num_kernels)

    def initialize(self):
        """
        See base class.
        """
        limit = math.sqrt(6 / (np.prod(self.input_shape[1:]) + self.num_kernels))
        kernels = np.array(np.random.uniform(-limit, limit,
                                             (self.num_kernels,
                                              self.d_kernel[0],
                                              self.d_kernel[1],
                                              self.input_shape[-1])),
                           dtype=np.float64)
        bias = np.zeros(self.num_kernels, dtype=np.float64)
        self.parameters = [placeholders.Variable(kernels), placeholders.Variable(bias)]
        self._build_graph()

    def _build_graph(self):
        """
        See base class.
        """
        if self.padding is True:
            p = self._calculate_pad_dims()
            zero_pad = convolution.ZeroPadding(self.input_node, p)
            conv = convolution.Convolution(zero_pad, self.parameters[0], self.parameters[1])
        else:  # self.padding is False
            conv = convolution.Convolution(self.input_node, self.parameters[0], self.parameters[1])
        act = ACTIVATION_REGISTRY[self.activation](conv)
        self.output_node = act

    def _calculate_pad_dims(self):
        return [(i - 1)/2 for i in self.d_kernel]


class FastConv(Conv):

    def _build_graph(self):
        if self.padding is True:
            p = self._calculate_pad_dims()
            node = convolution.ZeroPadding(self.input_node, p)
        else:
            node = self.input_node
        node = fast_convolution.FastConv(node, self.parameters[0], self.parameters[1])
        self.output_node = ACTIVATION_REGISTRY[self.activation](node)
