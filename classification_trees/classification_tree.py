"""
Within this file will live the main function for classifcation trees
"""
import pandas as pd


from classification_trees.node import Node

from typing import List


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
