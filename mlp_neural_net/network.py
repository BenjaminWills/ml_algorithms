import sys

# Add higher level folders for importing
sys.path.insert(0, "../")

from utility.logger import make_logger

import numpy as np

from layer import Layer, RANDOM_MULITPLIER

from copy import deepcopy
from typing import List, Callable, Union
from tqdm import tqdm

logger = make_logger(logging_path="./mlp-nn.log", save_logs=True, logger_name="MLPNN")

INCREMENT = 10e-5

Weights = List[np.array]
Weight_derivative = List[np.array]
Biases = List[np.array]
Bias_derivative = List[np.array]


def MSE(
    predictions: np.array,
    truth: np.array,
) -> float:
    num_inputs = len(predictions)
    return (1 / num_inputs) * (np.linalg.norm(truth - predictions))


class Network:
    def __init__(
        self,
        layer_depths: List[int],
        activations: Union[List[Callable], Callable],
        weights: List[np.array] = None,
        biases: List[np.array] = None,
    ) -> None:
        # Initialise layers
        self.layers = self.__initialise_layers(layer_depths, activations, biases)

        # Save number of layers
        self.num_layers = len(layer_depths)
        self.layer_depths = layer_depths

        # Initialise weights
        self.weights = weights or self.__initialise_weights(layer_depths)

        # Initialise biases
        self.biases = self.__get_biases()

        # Initialise activations
        self.activations = activations

    def __initialise_layers(
        self,
        layer_depths: List[int],
        activations: List[Callable],
        biases: Biases,
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
        if biases:
            # If biases is not none

            return [
                Layer(depth, activation_function, bias)
                for (depth, activation_function, bias) in zip(
                    layer_depths, activations, biases
                )
            ]
        else:
            # If biases is none then we randomly allocate biases
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
        layer_weights: Weights = []
        for index in range(self.num_layers - 1):
            """
            The weights have a dimension of layer_depths[index] x layer_depths[index + 1]
            and describe how each layer relates to the next.
            """
            current_depth = layer_depths[index]
            next_depth = layer_depths[index + 1]

            weights = np.random.rand(current_depth, next_depth) * RANDOM_MULITPLIER

            layer_weights.append(weights)

        return layer_weights

    def __get_biases(self) -> List[np.array]:
        """Gets the biases from the layers of the network.
        (Used in the backpropogation step)

        Returns
        -------
        List[np.array]
            List of bias vectors for each layer, the shape
            need not be uniform - dependent on the layer
            depth at that point.
        """
        layers = self.layers
        biases: Biases = [layer.biases for layer in layers]
        return biases

    def propogate_forwards(self, entries: np.array) -> None:
        """Propogates an input forwards through a network, this means that the weights are used to
        calculate the value for each node.

        Returns
        -------
        None
        """
        for index in range(self.num_layers):
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

                activation = self.activations[index]
                new_values = [activation(value) for value in new_values]

                # Set the current values to be the new values, and apply the activation function to them
                current_layer.values = new_values

                # Update the class variable.
                self.layers[index] = current_layer

    def propogate_backwards(self, cost_function: Callable) -> None:
        """Propogate backwards through the network to update the weights and biases.

        NOTE: going to brute force this one. Fix with algebra in future.

        Parameters
        ----------
        cost_function : Callable
            We expect our cost function to be a function of the weights, biases and the
            input of the function. i.e C = C(x, [W_1,...,W_n-1], [b_1,...,b_n]).
        """
        pass

    def calculate_bias_derivative(
        self, entries: np.array, truth: np.array
    ) -> Bias_derivative:
        """Calculates partial derivative w.r.t each bias in the network.

        Parameters
        ----------
        cost_function : Callable
            We expect our cost function to be a function of the weights, biases and the
            input of the function. i.e C = C(x, [W_1,...,W_n-1], [b_1,...,b_n]).

        Returns
        -------
        List[np.array]
            A list of all of the partial derivates in the network biases. This will be
            in the same order as the normal biases
        """
        # Propogate forwards and retrieve entries.
        self.propogate_forwards(entries)
        predictions = self.get_output()

        # Calculate oringinal MSE
        original_mse = MSE(predictions, truth)

        # Extract biases from the network
        network_biases = self.__get_biases()

        # Initialise bias derivatives
        bias_derivatives: List[np.array] = []

        for index in tqdm(range(self.num_layers)):
            biases = self.layers[index].biases
            dimension = biases.shape[0]

            # Initialise derivative vector
            layer_bias_derivative = np.zeros(dimension)

            # Get derivative
            for bias_index in range(dimension):

                # Create a new network with these biases.
                # We only want to update one element of the biases and keep the rest the same
                # That is to say that we want to update the index, bias_index'th element
                incremented_bias = deepcopy(network_biases)
                incremented_bias[index][bias_index] += INCREMENT

                incremented_network = Network(
                    self.layer_depths, self.activations, self.weights, incremented_bias
                )

                # Generate the new predicion
                incremented_network.propogate_forwards(entries)
                incremented_prediction = incremented_network.get_output()

                # Calculate new MSE
                new_mse = MSE(
                    incremented_prediction,
                    truth,
                )

                # Calculate partial derivative
                partial_derivative = (new_mse - original_mse) / INCREMENT

                # Update layer bias derivative
                layer_bias_derivative[bias_index] = partial_derivative

                # Remove the copy to save memory
                del incremented_bias

            bias_derivatives.append(layer_bias_derivative)
        return bias_derivatives

    def calculate_weight_derivtives(
        self, entries: np.array, truth: np.array
    ) -> Weight_derivative:
        pass

    def get_output(self) -> np.array:
        """Gets the output of the network.

        Returns
        -------
        np.array
            An array of the outputs of the network
        """
        return self.layers[-1].values
