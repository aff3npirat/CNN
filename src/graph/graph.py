from src.nodes import placeholders


class Graph:
    """
    Computational graph consisting of Nodes.

    Output of graph is output of last Node in graph.

    Attributes:
        ordered_nodes (list/tuple[Node,...]): All Nodes sorted, so that the output of a Node N, which comes after a Node
            M, is dependent on output of M.  Before computing N.output, M.output has to be computed.
    """

    def __init__(self, input_nodes):
        """
        Initializes ordered_nodes with all nodes in topological order.

        Args:
              input_nodes (list/tuple[Node,...]): all Inputs and Placeholders.
        """
        self.ordered_nodes = self.topological_sort(input_nodes)

    def topological_sort(self, input_nodes):
        """
        Returns a topological sort of graph using Kahn's algorithm.

        Args:
            input_nodes (list/tuple[Node,...]): Nodes to start.

        Returns:
            List of Nodes, representing graph in topological sort.
        """
        G = {}
        nodes = [n for n in input_nodes]
        while len(nodes) > 0:
            n = nodes.pop(0)
            if n not in G:
                G[n] = 0
            for m in n.output_nodes:
                if m not in G:
                    G[m] = 0
                    if m not in nodes:
                        nodes.append(m)
                G[m] += 1

        L = []
        S = set(input_nodes)
        while len(S) > 0:
            n = S.pop()
            L.append(n)
            for m in n.output_nodes:
                G[m] -= 1
                if G[m] == 0:
                    S.add(m)
        return L

    def compute(self):
        """
        Computes output values for all nodes in graph.
        """
        for node in self.ordered_nodes:
            node.compute()

    def backpass(self, grad_cost=None):
        """
        Computes gradients for all nodes in graph.
        """
        for node in self.ordered_nodes[::-1]:
            if isinstance(node, placeholders.GradPlaceholder):
                node.backpass(grad_cost)
            else:
                node.backpass()
