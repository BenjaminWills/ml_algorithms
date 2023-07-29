"""
Within this file will live the main function for classifcation trees
"""
import pandas as pd

from classification_trees.utility.binary_check import find_binary_columns
from classification_trees.node import Node
from classification_trees.data_structures.queue import Queue

from typing import List, Union


class Classification_tree:
    def __init__(
        self,
        data: pd.DataFrame,
        classification_column: str = None,
    ) -> None:
        self.data = data
        self.classification_column = classification_column or data.columns[-1]

        self.discrete_columns = find_binary_columns(self.data)

        self.root = Node(data=data, discrete_columns=self.discrete_columns)

        # Populate the tree
        self.populate_tree()

    def populate_tree(self) -> None:
        queue = Queue([self.root])

        while not queue.is_empty():
            node = queue.poll()
            node.find_nodes_children()

            # If the node is not a leaf
            if not node.leaf:
                left, right = node.left, node.right
                if left:
                    queue.add(left)
                if right:
                    queue.add(right)
        return None

    def get_nodes(self):
        # Gets nodes via a queue data structure
        queue = Queue([self.root])
        node_values = []
        while not queue.is_empty():
            node = queue.poll()
            column, value, left, right = (
                node.value_column,
                node.value,
                node.left,
                node.right,
            )
            node_values.append({"value": value, "column": column})
            if left:
                queue.add(left)
            if right:
                queue.add(right)
        return node_values

    def classify(self, input_data: Union[pd.Series, dict]) -> int:
        """Classifies a row of data based on the features that it was
        trained on.

        Parameters
        ----------
        input_data : Union[pd.Series, dict]
            Data that matches the style of the training data

        Returns
        -------
        int
            0 or 1
        """
        prediction = None

        node = self.root

        while True:
            prediction = node.predict(input_data)
            if prediction == True:
                node = node.left
            if prediction == False:
                node = node.right
            if node.leaf:
                classification = list(node.data[self.classification_column])[0]
                break
        return classification
