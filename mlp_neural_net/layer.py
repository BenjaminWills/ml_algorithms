import numpy as np

from node import Node

from typing import Callable


class Layer:
    """
    A layer is a collection of nodes.
    """

    def __init__(self, depth: int, activation: Callable, biases: np.array) -> None:
        self.activation = activation
        self.biases = np.random.rand(depth, 1) * 1_000  # A depth dimensional vector.
        self.nodes = [Node(bias) for bias in biases]
