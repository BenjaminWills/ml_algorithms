import pandas as pd

from typing import List, Union


def is_binary(column: pd.Series) -> bool:
    column = column.astype(int)

    mapped_col = set(column)
    if mapped_col == {0, 1}:
        return True
    else:
        return False


def find_binary_columns(data: pd.DataFrame) -> List[Union[int, str]]:
    binary_cols = []
    for column in data.columns:
        try:
            if is_binary(data[column]):
                binary_cols.append(column)
        except:
            pass
    return binary_cols
