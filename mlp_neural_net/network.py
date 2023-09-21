import sys

# Add higher level folders for importing
sys.path.insert(0, "../")

from utility.logger import make_logger

import numpy as np

from layer import Layer

from typing import List, Callable, Union
from tqdm import tqdm

logger = make_logger(logging_path="./mlp-nn.log", save_logs=True, logger_name="MLPNN")


class Network:
    def __init__(
        self, layer_depths: List[int], activations: Union[List[Callable], Callable]
    ) -> None:
        # Initialise layers
        self.layers = self.__initialise_layers(layer_depths, activations)

        # Save number of layers
        self.num_layers = len(layer_depths)

        # Initialise weights
        self.weights = self.__initialise_weights(layer_depths)

        # Initialise activations
        self.actications = activations

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
        for index in range(self.num_layers - 1):
            """
            The weights have a dimension of layer_depths[index] x layer_depths[index + 1]
            and describe how each layer relates to the next.
            """
            current_depth = layer_depths[index]
            next_depth = layer_depths[index + 1]

            weights = np.random.rand(current_depth, next_depth) * 1_000

            layer_weights.append(weights)

        return layer_weights

    def propogate_forwards(self, entries: np.array) -> None:
        """Propogates an input forwards through a network, this means that the weights are used to
        calculate the value for each node.

        Returns
        -------
        None
        """
        for index in tqdm(range(self.num_layers)):
            """
            At each layer we want to calcualte the value for each node.
            """
            if index == 0:
                # The values for the very first nodes are simply the inputted values.
                first_layer = self.layers[index]
                first_layer.values = entries
                self.layers[index] = first_layer

            else:
                # Current layer information
                current_layer = self.layers[index]
                biases = current_layer.biases

                # Previous layer information
                previous_layer = self.layers[index - 1]
                previous_values = previous_layer.values
                weights = self.weights[index - 1]

                # Calculation of the new values for the layer
                weights_transpose = np.transpose(weights)
                new_values = np.matmul(weights_transpose, previous_values) + biases

                # Set the current values to be the new values
                current_layer.values = new_values

                # Update the class variable.
                self.layers[index] = current_layer

        logger.info("Fed forwards.")
