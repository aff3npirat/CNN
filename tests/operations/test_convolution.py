import numpy as np
import unittest

from src.nodes import placeholders
from src.nodes.operations import convolution


class TestConvolution(unittest.TestCase):

    def test_compute_with_symmetrical_input(self):
        # given
        x = placeholders.Variable(np.random.rand(4, 11, 11, 3))
        w = placeholders.Variable(np.random.rand(16, 3, 3, 3))
        b = placeholders.Variable(np.random.rand(16))

        # when
        node = convolution.Convolution(x, w, b)
        node.compute()

        # then
        assert node.output.shape == (4, 9, 9, 16)
        expected_val = np.sum(w.output[1, :, :, :] * x.output[0, 0:3, 0:3, :]) + b.output[1]
        assert abs(node.output[0, 0, 0, 1] - expected_val) < 1e-8
        expected_val = np.sum(w.output[15, :, :, :] * x.output[3, 8:11, 2:5, :]) + b.output[15]
        assert abs(node.output[3, 8, 2, 15] - expected_val) < 1e-8

    def test_compute_with_asymmetrical_input(self):
        # given
        x = placeholders.Variable(np.random.rand(4, 11, 9, 3))
        w = placeholders.Variable(np.random.rand(16, 5, 3, 3))
        b = placeholders.Variable(np.random.rand(16))

        # when
        node = convolution.Convolution(x, w, b)
        node.compute()

        # then
        assert node.output.shape == (4, 7, 7, 16)
        expected_val = np.sum(w.output[2, :, :, :] * x.output[0, 0:5, 0:3, :]) + b.output[2]
        assert abs(node.output[0, 0, 0, 2] - expected_val) < 1e-8

    def test_backpass_only_shape(self):
        # given
        x = placeholders.Variable(np.random.rand(4, 5, 5, 3))
        w = placeholders.Variable(np.random.rand(16, 3, 3, 3))
        b = placeholders.Variable(np.random.rand(16))

        # when
        node = convolution.Convolution(x, w, b)
        placeholders.GradPlaceholder(node).backpass(np.ones((4, 3, 3, 16)))
        node.backpass()

        # then
        assert node.gradients[x].shape == (4, 5, 5, 3)
        assert node.gradients[w].shape == (16, 3, 3, 3)
        assert node.gradients[b].shape == (16,)

    def test_backpass_with_single_feature_and_kernel(self):
        # given
        x = placeholders.Variable(np.random.rand(1, 5, 5, 3))
        w = placeholders.Variable(np.random.rand(1, 3, 3, 3))
        b = placeholders.Variable(np.random.rand(1))
        node = convolution.Convolution(x, w, b)

        # when
        upstream_grad = np.array([[[[0], [1], [2]], [[3], [0.5], [5]], [[-1], [6], [-4]]]])
        placeholders.GradPlaceholder(node).backpass(upstream_grad)
        node.backpass()

        # then
        expected_val = 0.0
        for j in range(3):
            for i in range(2):
                expected_val += upstream_grad[0, i + 1, j, 0] * w.output[0, 2 - i, 2 - j, 0]
        assert abs(node.gradients[x][0, 3, 2, 0] - expected_val) < 1e-8
        expected_val = 0.0
        for j in range(3):
            for i in range(3):
                expected_val += upstream_grad[0, i, j, 0] * x.output[0, 1 + i, 1 + j, 0]
        assert abs(node.gradients[w][0, 1, 1, 0] - expected_val) < 1e-8
        assert abs(node.gradients[b][0] - np.sum(upstream_grad)) < 1e-8

    def test_backpass_with_multiple_feature_and_kernel(self):
        # given
        x = placeholders.Variable(np.random.rand(4, 5, 5, 3))
        w = placeholders.Variable(np.random.rand(5, 3, 3, 3))
        b = placeholders.Variable(np.random.rand(5))
        node = convolution.Convolution(x, w, b)

        # when
        upstream_grad = np.random.rand(4, 3, 3, 5)
        placeholders.GradPlaceholder(node).backpass(upstream_grad)
        node.backpass()

        # then
        expected_val = 0.0
        for v in range(5):
            for j in range(3):
                for i in range(2):
                    expected_val += upstream_grad[0, i + 1, j, v] * w.output[v, 2 - i, 2 - j, 0]
        assert abs(node.gradients[x][0, 3, 2, 0] - expected_val) < 1e-8
        expected_val = 0.0
        for v in range(4):
            for j in range(3):
                for i in range(3):
                    expected_val += upstream_grad[v, i, j, 0] * x.output[v, 1 + i, 1 + j, 1]
        expected_val /= 4
        assert abs(node.gradients[w][0, 1, 1, 1] - expected_val) < 1e-8
        assert (abs(node.gradients[b] - np.sum(upstream_grad, axis=(0, 1, 2))/4) < 1e-8).all()

    def test_convolve_with_convolve_iterative(self):
        # given
        x = np.random.rand(10, 32, 32, 3)
        w = np.random.rand(32, 3, 3, 3)
        b = np.random.rand(32)

        # then
        assert (abs(convolution.convolve(x, w, b) - convolution.convolve_iterative(x, w, b)) < 1e-8).all()


class TestZeroPadding(unittest.TestCase):

    def test_compute_symmetrical(self):
        # given
        x = placeholders.Variable(np.arange(2000).reshape(4, 10, 10, 5))
        node = convolution.ZeroPadding(x, (2, 2))

        # when
        node.compute()

        # then
        assert node.output.shape == (4, 14, 14, 5)
        assert np.sum(node.output) == np.sum(x.output)
        assert (node.output[:, 2:12, 2:12, :] == x.output).all()

    def test_compute_asymmetrical(self):
        # given
        x = placeholders.Variable(np.arange(2000).reshape(4, 10, 10, 5))
        node = convolution.ZeroPadding(x, (2, 3))

        # when
        node.compute()

        # then
        assert node.output.shape == (4, 14, 16, 5)
        assert np.sum(node.output) == np.sum(x.output)
        assert (node.output[:, 2:12, 3:13, :] == x.output).all()

    def test_backpass_symmetrical(self):
        # given
        x = placeholders.Variable(np.arange(2000).reshape(4, 10, 10, 5))
        node = convolution.ZeroPadding(x, (2, 2))

        # when
        placeholders.GradPlaceholder(node).backpass(np.ones((4, 14, 14, 5), dtype=np.int))
        node.backpass()

        # then
        assert node.gradients[x].shape == (4, 10, 10, 5)
        assert (node.gradients[x] == np.ones((4, 10, 10, 5))).all()

    def test_backpass_asymmetrical(self):
        # given
        x = placeholders.Variable(np.arange(2000).reshape(4, 10, 10, 5))
        node = convolution.ZeroPadding(x, (2, 4))

        # when
        placeholders.GradPlaceholder(node).backpass(np.ones((4, 14, 18, 5), dtype=np.int))
        node.backpass()

        # then
        assert node.gradients[x].shape == (4, 10, 10, 5)
        assert (node.gradients[x] == np.ones((4, 10, 10, 5))).all()

