class Node:
    """
    A Node in a computational graph.

    Attributes:
        input_nodes (list[Node,...]): All input nodes.
        output_nodes (list[Node,...]): All nodes that take this node as input.
        output (): Result of compute method, can be any type, but should be compatible with output nodes.
        gradients (dict[Node:]): Stores for each input node gradient of graph-output (L) wrt input node, so that for
            input node x gradients[x]= dL/dx.
    """

    def __init__(self, input_nodes=[]):
        """
        Initializes Node with values and sets this node as output node of all input nodes.

        Args:
            input_nodes (list[Node,...]): List which holds all input nodes.
        """
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in self.input_nodes:
            node.output_nodes.append(self)

        self.output = None
        self.gradients = {}

    def compute(self):
        """
        Logic for operation, result should be stored in self.output.
        """
        raise NotImplementedError

    def backpass(self):
        """
        Computes gradient of output wrt inputs.

        One forwardpass must be done before method call.
        """
        raise NotImplementedError
