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

        self.nodes = [Node(bias) for bias in self.__initialise_biases()]

    def __initialise_biases(self) -> np.array:
        biases = np.random.rand(self.depth, 1) * 1_000  # A depth dimensional vector.
        return biases
