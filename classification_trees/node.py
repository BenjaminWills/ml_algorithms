import pandas as pd

from collections import defaultdict
from typing import List

"""
When we have a node on our tree we want to know the following things:
    - What value it represents
    - Who it's children are
    - What data it represents
    - What it's information entropy is
"""

from classification_trees.utility.split_data import (
    get_classification_proportions,
    split_data_on_float,
)
from classification_trees.entropy_functions.entropy import log_entropy
from classification_trees.utility.information_gain import calculate_information_gain


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

    def find_nodes_children(self, discrete_columns: List[str] = []) -> List[str]:
        """
        We want to go through each unique row of the dataframe and find the
        entropy when the dataframes are split.
        """
        data = self.data

        # ASSUMPTION: the classification column is the final one, change later
        # being lazy atm.
        non_classification_columns = data.columns[:-1]

        # Define a dictionaty whose default elements are lists
        unique_col_values = defaultdict(lambda: [])

        for column in non_classification_columns:
            unique_col_values[column].extend(data[column].unique())

        unique_col_values = dict(unique_col_values)

        # Find data split that maximises information gain

        max_information_gain = -1
        max_value = 0
        max_column_name = ""

        for column, values in unique_col_values.items():

            # Discrete columns are dealt with differently
            if column in discrete_columns:
                is_discrete = True
            else:
                is_discrete = False

            for value in values:
                split = split_data_on_float(
                    data, column, value, is_discrete=is_discrete
                )
                information_gain = calculate_information_gain(
                    self.entropy, [split["top"], split["bottom"]]
                )
                if information_gain > max_information_gain:
                    max_information_gain = information_gain
                    max_value = value
                    max_column_name = column
        return {"column": max_column_name, "value": max_value}
