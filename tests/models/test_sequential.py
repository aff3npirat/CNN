import numpy as np
import unittest

from src.nodes.operations import cast
from src.layers import layers
from src.models import sequential


class TestSequential(unittest.TestCase):

    def test_add(self):
        # given
        model = sequential.SequentialModel()
        layer1 = layers.Dense(5, input_shape=(5,))
        layer2 = layers.Dense(7)

        # when
        model.add(layer1)
        model.add(layer2, "name", np.float64)

        # then
        assert layer1.name == "Dense_1"
        assert layer2.name == "name"
        assert isinstance(layer2.output_node, cast.CastForwardOnly)
        assert layer2.output_node._dtype == np.float64



