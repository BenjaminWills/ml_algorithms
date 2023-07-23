import pandas as pd

"""
When we have a node on our tree we want to know the following things:
    - What value it represents
    - Who it's children are
    - What data it represents
    - What it's information entropy is
"""

from classification_trees.utility.split_data import get_classification_proportions
from classification_trees.entropy_functions.entropy import log_entropy


class Node:
    def __init__(
        self,
        data: pd.DataFrame,
        child_1=None,
        child_2=None,
        value_column: str = None,
        value: float = None,
        classification_column: str = None,
    ) -> None:
        # Subset of data that exists in the node
        self.data = data

        # Children of the node
        self.child_1 = child_1
        self.child_2 = child_2

        # Calculate entropy
        self.classification_column = classification_column or data.columns[-1]
        self.entropy = self.__calculate_node_entropy()

        # Save the value that the node represents as well as the column for which
        # the value belongs to
        self.value = value
        self.value_column = value_column

    def __calculate_node_entropy(self) -> float:
        classification_proportions = get_classification_proportions(
            self.data, self.classification_column
        )
        return log_entropy(classification_proportions)
