import numpy as np
import unittest

from src.nodes import placeholders
from src.nodes.operations import add


class TestAdd(unittest.TestCase):

    def test_backpass(self):
        # given
        a = placeholders.Variable(np.arange(10))
        b = placeholders.Variable(np.arange(10))

        # when
        node = add.Add(a, b)
        placeholders.GradPlaceholder(node).backpass(np.arange(10))
        node.backpass()

        # then
        assert (node.gradients[a] == np.arange(10)).all()
        assert (node.gradients[b] == np.arange(10)).all()
