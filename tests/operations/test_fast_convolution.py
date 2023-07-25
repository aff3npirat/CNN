import numpy as np
import time
import unittest

from src.nodes import placeholders
from src.nodes.operations import convolution, fast_convolution


class TestFastConvolution(unittest.TestCase):

    def test_compute(self):
        # given
        x = placeholders.Variable(np.random.rand(10, 32, 32, 4))
        w = placeholders.Variable(np.random.rand(16, 5, 3, 4))
        b = placeholders.Variable(np.random.rand(16))
        node = fast_convolution.FastConv(x, w, b)

        # when
        node.compute()

        # then
        assert node.output.shape == (10, 28, 30, 16)
        assert (abs(node.output - convolution.convolve(x.output, w.output, b.output)) < 1e-8).all()

    def test_backpass(self):
        # given
        x = placeholders.Variable(np.random.rand(10, 32, 32, 4))
        w = placeholders.Variable(np.random.rand(16, 5, 3, 4))
        b = placeholders.Variable(np.random.rand(16))
        node = fast_convolution.FastConv(x, w, b)
        conv = convolution.Convolution(x, w, b)

        # when
        node.compute()
        conv.compute()
        placeholders.GradPlaceholder(node).backpass(np.ones((10, 28, 30, 16)))
        placeholders.GradPlaceholder(conv).backpass(np.ones((10, 28, 30, 16)))
        node.backpass()
        conv.backpass()

        # then
        assert node.gradients[x].shape == (10, 32, 32, 4)
        assert node.gradients[w].shape == (16, 5, 3, 4)
        assert node.gradients[b].shape == (16,)
        assert (abs(node.gradients[x] - conv.gradients[x]) < 1e-8).all()
        assert (abs(node.gradients[w] - conv.gradients[w]) < 1e-8).all()
        assert (abs(node.gradients[b] - conv.gradients[b]) < 1e-8).all()

    def test_time(self):
        x = placeholders.Variable(np.random.rand(10, 32, 32, 4))
        w = placeholders.Variable(np.random.rand(16, 5, 3, 4))
        b = placeholders.Variable(np.random.rand(16))
        fast_conv = fast_convolution.FastConv(x, w, b)
        conv = convolution.Convolution(x, w, b)

        conv_time = time.perf_counter()
        conv.compute()
        conv_time = time.perf_counter() - conv_time

        fast_conv_time = time.perf_counter()
        fast_conv.compute()
        fast_conv_time = time.perf_counter() - fast_conv_time

        print(f"[Forward] Conv: {conv_time*1000:.3f}ms - FastConv: {fast_conv_time*1000:.3f}ms - factor: {conv_time/fast_conv_time:.2f}")

        placeholders.GradPlaceholder(conv).backpass(np.ones_like(conv.output))
        placeholders.GradPlaceholder(fast_conv).backpass(np.ones_like(fast_conv.output))

        conv_time = time.perf_counter()
        conv.backpass()
        conv_time = time.perf_counter() - conv_time

        fast_conv_time = time.perf_counter()
        fast_conv.backpass()
        fast_conv_time = time.perf_counter() - fast_conv_time

        print(f"[Backpass] Conv: {conv_time*1000:.3f}ms - FastConv: {fast_conv_time*1000:.3f}ms - factor: {conv_time/fast_conv_time:.2f}")

