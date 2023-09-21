import numpy as np

from layer import Layer

from typing import List, Callable


class Network:
    def __init__(self, layer_depths: List[int], activations: List[Callable]) -> None:
        # Initialise layers
        self.layers = self.__initialise_layers(layer_depths, activations)

        # Initialise weights
        self.weights = []
        for index in range(layer_depths - 1):
            pass

    def __initialise_layers(
        self, layer_depths: List[int], activations: List[Callable]
    ) -> List[Layer]:
        """Initialises the layers of the network.

        Parameters
        ----------
        layer_depths : List[int]
            The depths of each layer.
        activations : List[Callable]
            The activation functions that exist within each layer

        Returns
        -------
        List[Layer]
            A list of neural network layers
        """
        return [
            Layer(depth, activation_function)
            for (depth, activation_function) in zip(layer_depths, activations)
        ]

    def __initialise_weights(self, layer_depths: List[int]) -> List[np.array]:
        pass
