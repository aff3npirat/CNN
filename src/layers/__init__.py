from src.nodes.activations import binary, relu, softmax


ACTIVATION_REGISTRY = {
                       'relu': relu.Relu,
                       'softmax': softmax.Softmax,
                       'binary': binary.Binary,
                      }
