"""
Within this file will live the main function for classifcation trees
"""
import pandas as pd

from classification_trees.node import Node
from classification_trees.utility.split_data import get_classification_proportions
from classification_trees.entropy_functions.entropy import log_entropy


class Classification_tree:
    def __init__(self, data: pd.DataFrame, classification_column: str = None) -> None:
        self.data = data
        self.classification_column = classification_column or data.columns[-1]
        self.root = Node(data=data)

    def find_first_children():
        """
        We want to go through each unique row of the dataframe and find the
        entropy when the dataframes are split.
        """
        pass
