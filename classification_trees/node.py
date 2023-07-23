import pandas as pd


class Node:
    def __init__(
        self,
        data: pd.DataFrame,
        child_1=None,
        child_2=None,
        entropy: float = 0,
        value: float = None,
    ) -> None:
        self.data = data
        self.child_1 = child_1
        self.child_2 = child_2
        self.entropy = entropy
