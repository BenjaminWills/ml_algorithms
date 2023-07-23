"""
Within this file will live the main function for classifcation trees
"""
import pandas as pd


from classification_trees.node import Node
from classification_trees.data_structures.queue import Queue

from typing import List, Union


class Classification_tree:
    def __init__(
        self,
        data: pd.DataFrame,
        classification_column: str = None,
        discrete_columns: List[str] = [],
    ) -> None:
        self.data = data
        self.classification_column = classification_column or data.columns[-1]
        self.root = Node(data=data, discrete_columns=discrete_columns)

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

    def classify(row: Union[pd.Series, dict]) -> int:
        pass
