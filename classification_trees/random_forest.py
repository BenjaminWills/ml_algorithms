import pandas as pd

from classification_trees.classification_tree import Classification_tree

from typing import List, Union


class Random_forest:
    """The random forest classifier algorithm, pooling results from many
    classification trees.
    """

    def __init__(self, data: pd.DataFrame, num_trees: int) -> None:
        self.data = data
        self.num_trees = num_trees
        self.shuffled_percentage = 0.8

        self.trees = self.__create_forest()

    def __create_forest(self) -> List[Classification_tree]:
        """Creates the forest of trees, each trained on a random
        subset of the input data.

        Returns
        -------
        List[Classification_tree]
            A list of classification trees.
        """
        trees = []
        for tree in range(self.num_trees):
            # shuffle data
            shuffled_data = self.data.sample(frac=self.shuffled_percentage)
            tree = Classification_tree(shuffled_data)
            trees.append(tree)
        return trees

    def classify(self, input_data: Union[pd.Series, dict]) -> int:
        """Classifies the input data (which must share the structure of the training data)
        by finding the most common answer from the trees.

        Parameters
        ----------
        input_data : Union[pd.Series, dict]

        Returns
        -------
        int
            The predicted class
        """
        classifications = []
        for tree in self.trees:
            classifications.append(tree.classify(input_data))
        classifications = pd.Series(classifications)
        return classifications.value_counts().argmax()
