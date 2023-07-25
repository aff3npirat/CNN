import numpy as np
import time

from datetime import datetime
from src.graph import graph
from src.nodes import placeholders
from src.nodes.operations import cast
from src.utils import core, metrics


METRIC_REGISTRY = {
                   'categorical_acc': metrics.softmax_accuracy,
                   'accuracy': metrics.accuracy,
                  }


class SequentialModel:
    """
    Class representing a neural network.

    A Model consists of layers which have to be subclasses of class layers.Layer and a computational graph.

    Attributes:
        layers (list[Layer,...]): Stores all layers of model.
        graph (Graph): The computational graph, computing model's output.
        loss (Node): The loss function used.
        optimizer (Optimizer): The algorithm used for updating layer parameters.
        max_grad_val (number): Maximum value for gradients.
        max_param_val (number): Maximum value for parameters.
        history (Dict[String: list): History of different values.
            Can be { "test_acc", "test_loss", "train_acc", "train_loss", "update_ratio" }
    """

    def __init__(self):
        self.layers = []
        self.graph = None
        self.loss = None
        self.optimizer = None
        self.metric = None
        self._train_loss = []
        self._train_acc = []
        self._test_loss = []
        self._test_acc = []
        self._update_ratio = {}
        self.max_grad_val = np.inf
        self.max_param_val = np.inf

    @property
    def history(self):
        return {
                "train_loss": self._train_loss,
                "train_acc": self._train_acc,
                "test_loss": self._test_loss,
                "test_acc": self._train_acc,
                "update_ratio": self._update_ratio,
               }

    @property
    def input_node(self):
        return self.layers[0].input_node

    @property
    def output_node(self):
        return self.layers[-1].output_node

    @output_node.setter
    def output_node(self, node):
        self.layers[-1].output_node = node

    def add(self, layer, name=None, dtype=None):
        """
        Adds layer to model.

        Assigns input_shape, input_node to output_shape, output_node of previous layer and initializes layer by
        calling respective method.

        Args:
            layer (Layer): Layer to be added.
            name (string): Optional; Name added layer should get, if None layer gets name consisting of class name and
                index.
            dtype (Type): Data-type layers output will be cast to.  Set to None for no casting.
        """
        if name is None:
            name = type(layer).__name__ + "_" + str(len(self.layers) + 1)
        layer.name = name

        if len(self.layers) >= 1:
            layer.input_node = self.output_node
            layer.input_shape = self.layers[-1].output_shape()
        else:
            layer.input_node = placeholders.Variable()
        layer.initialize()
        self.layers.append(layer)

        if dtype is not None:
            self.output_node = cast.CastForwardOnly(layer.output_node, dtype)

    def compile(self, loss, optimizer, metric):
        """
        Creates computational graph.

        Args:
            loss (Loss): Loss function used.
            optimizer (Optimizer): Optimizer used.
            metric (string): Metric evaluated while training and testing.
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metric = METRIC_REGISTRY.get(metric)

        self.output_node = placeholders.GradPlaceholder(self.output_node)

        input_nodes = [self.input_node]
        for layer in self.layers:
            input_nodes += layer.parameters
        self.graph = graph.Graph(input_nodes)

    def _forward_pass(self, x):
        self.input_node.output = x
        self.graph.compute()

    def _backpass(self, y_hat, y):
        self.graph.backpass(self.loss.gradient(y_hat, y))

    def prediction(self, x):
        """
        Makes a prediction on input_ and calculates loss.

        Args:
            x (numpy.array): Examples used for prediction, should be of shape MxN, with N equal to input_shape of first
                layer.  And M equals number of examples, M can be omitted.

        Returns:
            Numpy.array with shape MxK, with K equal to output_shape of last layer.  K-array at index i is prediction
            on ith example (input[i]).  If M was omitted for input_, output will have shape 1xK.

        Raises:
            ValueError: If input_ has invalid shape.
        """
        d_x = x.shape
        d_in = self.layers[0].input_shape
        if d_in == d_x:
            x = x.reshape(1, *x.shape)
        elif d_x[1:] != d_in:
            raise ValueError("Wrong input shape, expected " + str(self.layers[0].input_shape) + " got " + str(x.shape))

        self._forward_pass(x)
        return self.output_node.output

    def fit(self, x_train, y_train, x_test=None, y_test=None, epochs=1, batch_size=None, verbose=True, fname=None,
            save_update_ratio=False, train_tol=0.99, test_tol=0.99):
        """
        Makes a prediction and updates parameters.

        Args:
            x_train (numpy.array): Input values used for training.
            y_train (Array): Target values used for training, should be compatible with loss function.
            x_test (numpy.array): Optional; Input values used for testing.
            y_test (Array) Optional; Target values used for testing, should be compatible with loss function.
            epochs (int): Number of epochs used during training
            batch_size (int): Optional; Size of batch used during training.
            verbose (bool): Optional; If True, after each epoch loss, accuracy are printed.
            fname (string): Optional; Name of file to save to.
            save_update_ratio (bool): Optional; If True saves ratio of parameter updates in model history.
            train_tol (float or None): Optional; Threshold for train accuracy to terminate.  If test data is given train_tol is
                ignored.
            test_tol (float or None): Opional; Threshold for test accuracy to terminate.

        Returns:
            Prediction made.
        """
        validation = x_test is None or y_test is None
        if validation:
            test_tol = None
        else:
            train_tol = None

        m = x_train.shape[0]
        if batch_size is None or batch_size > m:
            batch_size = 1

        x_batches = core.generate_batches(x_train, batch_size)
        y_batches = core.generate_batches(y_train, batch_size)
        for epoch in range(epochs):
            if verbose:
                print(datetime.today().strftime(f"%H:%M Epoch {epoch + 1}/{epochs} - "), end="", flush=True)
            epoch_start = time.time()

            y_hat = np.zeros((m, *self.layers[-1].output_shape()))
            for idx, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
                self._forward_pass(x_batch)
                y_hat_batch = self.output_node.output
                y_hat[idx * batch_size:(idx * batch_size) + y_hat_batch.shape[0]] = y_hat_batch

                self._backpass(y_hat_batch, y_batch)
                self.update(save_update_ratio)
            self._train_loss.append(self.loss.loss(y_hat, y_train))
            self._train_acc.append(self.metric(y_hat, y_train))

            if not validation:
                y_hat = self.prediction(x_test)
                self._test_loss.append(self.loss.loss(y_hat, y_test))
                self._test_acc.append(self.metric(y_hat, y_test))
            epoch_time = time.time() - epoch_start

            if (test_tol is not None and self._test_acc[-1] >= test_tol) or (train_tol is not None and self._train_acc[-1] >= train_tol):
                idx = -1 if fname is None else fname.rfind('/')
                if 0 <= idx < len(fname)-1:
                    path = fname[:idx + 1]
                    self.save_to_file(path, fname[idx+1:])
                    print(f"Saved to file '{path}{fname[idx+1:]}'")
                return

            if verbose is True:
                print(f"{int(epoch_time)}s")
                if validation:
                    print(f"      loss: {self._train_loss[-1]:.5f} - accuracy: {self._train_acc[-1]:.2f}")
                else:
                    print(f"      loss: {self._train_loss[-1]:.5f} - accuracy: {self._train_acc[-1]:.2f} | "
                          f"val_loss: {self._test_loss[-1]:.5f} - val_accuracy: {self._test_acc[-1]:.2f}")

        idx = -1 if fname is None else fname.rfind('/')
        if 0 <= idx < len(fname) - 1:
            path = fname[:idx + 1]
            self.save_to_file(path, fname[idx + 1:])
            print(f"Saved to file '{path}{fname[idx + 1:]}'")

    def update(self, save_update_ratio=False):
        """
        Updates all parameters.
        """
        trainables = self._get_params_as_list()

        for param in trainables:
            param.gradients[param] = np.minimum(param.gradients[param], self.max_grad_val)

        if save_update_ratio is True:
            for param in trainables:
                param_scale = np.linalg.norm(param.output.ravel())
                update_scale = np.linalg.norm(param.gradients[param].ravel())
                update_ratio = update_scale / param_scale
                if param in self._update_ratio.keys():
                    self._update_ratio[param].append(update_ratio)
                else:
                    self._update_ratio[param] = [update_ratio]

        self.optimizer.update(trainables)

        for param in trainables:
            param.output = np.minimum(param.output, self.max_param_val)

    def summary(self):
        """
        Prints a summary of model.
        """
        line_length = 45
        print("#: layer_type     output_shape     parameters")
        print("=" * line_length)

        def print_layer(index, layer, num_param, separating_line=True):
            filling_whitespaces = (" "*(15 - len(layer.name) - (1 if index > 9 else 0)),
                                   " "*(17 - len(str(layer.output_shape()))),
                                   " "*(10 - len(str(num_param))))
            print("{}: {}{}{}".format(str(index),
                                      layer.name + filling_whitespaces[0],
                                      str(layer.output_shape()) + filling_whitespaces[1],
                                      str(num_param) + filling_whitespaces[2]))
            if separating_line:
                print("-" * line_length)

        model_params = 0
        for i in range(len(self.layers)):
            layer_params = 0
            for param in self.layers[i].parameters:
                layer_params += np.prod(param.output.shape)
            print_layer(i, self.layers[i], layer_params, i==len(self.layers[:-1]))
            model_params += layer_params
        print("=" * line_length)
        print("Total Parameters: " + str(model_params))

    def save_to_file(self, path, fname):
        param_ids = ['w', 'b']
        weights = {}
        params = self._get_params_as_dict()
        for key in params.keys():
            for i in range(len(params[key])):
                weights[key + "_" + param_ids[i]] = params[key][i].output
        np.savez(path + fname, **weights)

    def _get_params_as_list(self):
        params = []
        for layer in self.layers:
            params += layer.parameters
        return params

    def _get_params_as_dict(self):
        return {layer.name: layer.parameters for layer in self.layers}

    def get_weights(self, as_list=True):
        if as_list is True:
            return [p.output for p in self._get_params_as_list()]
        else:
            weights = {}
            params = self._get_params_as_dict()
            for key in params:
                weights[key] = [p.output for p in params[key]]
            return weights

    def set_weights(self, new_weights, offset=0):
        """
        Args:
            new_weights (array[np.ndarray]): Array-like containing new values for weights.
            offset (int): Optional; Index to start from.
        """
        params = self._get_params_as_list()
        for i in range(len(params)):
            params[i + offset].output = new_weights[i + offset]

