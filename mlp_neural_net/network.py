import numpy as np

from layer import Layer

from typing import List, Callable


class Network:
    def __init__(self, layer_depths: List[int], activations: List[Callable]) -> None:
        # Initialise layers
        self.layers = self.__initialise_layers(layer_depths, activations)

        # Initialise weights
        self.weights = self.__initialise_weights(layer_depths)

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
        """Initialises the weights between the layers.

        Parameters
        ----------
        layer_depths : List[int]
            A list of the depths of the layers

        Returns
        -------
        List[np.array]
            A list of weights
        """
        layer_weights = []
        for index in range(layer_depths - 1):
            """
            The weights have a dimension of layer_depths[index] x layer_depths[index + 1]
            and describe how each layer relates to the next.
            """
            current_depth = layer_depths[index]
            next_depth = layer_depths[index + 1]

            weights = np.random.rand(current_depth, next_depth)

            layer_weights.append(weights)

        return layer_weights
