import numpy as np

from node import Node

from typing import Callable


class Layer:
    """
    A layer is a collection of nodes.
    """

    def __init__(self, depth: int, activation: Callable) -> None:
        self.activation = activation
        self.depth = depth

        self.biases = self.__initialise_biases()
        self.nodes = [Node(bias) for bias in self.biases]
        self.values = [node.value for node in self.nodes]

    def __initialise_biases(self) -> np.array:
        biases = np.random.rand(self.depth) * 1_000  # A depth dimensional vector.
        return biases
