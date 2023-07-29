import pandas as pd

from classification_trees.classification_tree import Classification_tree

from typing import List


class Random_forest:
    def __init__(self, data: pd.DataFrame, num_trees: int) -> None:
        self.data = data
        self.num_trees = num_trees
        self.shuffled_percentage = 0.8

        self.trees = self.__create_forest()

    def __create_forest(self) -> List[Classification_tree]:
        trees = []
        for tree in range(self.num_trees):
            # shuffle data
            shuffled_data = self.data.sample(frac=self.shuffled_percentage)
            tree = Classification_tree(shuffled_data)
            trees.append(tree)
        return trees

    def classify(self, input_data):
        classifications = []
        for tree in self.trees:
            classifications.append(tree.classify(input_data))
        classifications = pd.Series(classifications)
        return classifications.value_counts().argmax()
