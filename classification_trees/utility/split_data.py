import pandas as pd

from typing import Union, Dict, List

from classification_trees.entropy_functions.entropy import log_entropy


def get_classification_proportions(
    data: pd.DataFrame, classification_col: str
) -> List[float]:
    # This function will return the number of 1's relative to the number of 0's.

    if len(data) == 0:
        return [0.5, 0.5]  # This will maxmise entropy!

    num_ones = data[classification_col].value_counts().to_dict().get(1, 0)
    ones_proportions = num_ones / len(data)

    return [ones_proportions, 1 - ones_proportions]


def split_data_on_float(
    data: pd.DataFrame,
    column: str,
    value: float,
    classification_col: Union[None, str] = None,
    is_discrete: bool = False,
) -> Dict[str, Dict[str, Union[pd.DataFrame, float]]]:
    # Default value of classification column will be the last column of the dataframe
    if classification_col is None:
        classification_col = data.columns[-1]

    # This function will split some data based on a numeric value.

    # BUG: this bug occurs due to the binary nature of the columns. We must request equality rather than an innequality.
    # The nuance with this is for continuous columns such as age, these would have to be one-hot encoded which could be
    # computationally expensive. A workaround is having a discrete argument to append to the mask.

    mask = data[column] <= value
    if is_discrete:
        mask = data[column] == value

    # Splits the data based on the inequality
    bottom = data[mask]
    top = data[~mask]

    # Get relative proportions of parent dataset
    total_rows = len(data)

    bottom_weight = len(bottom) / total_rows
    top_weight = len(top) / total_rows

    # Get proportions of classifications
    # NOTE: for now we assume that the classification is binary, for the entropy calculation.

    bottom_entropy_proportion = get_classification_proportions(
        bottom, classification_col
    )
    top_entropy_proportion = get_classification_proportions(top, classification_col)

    bottom_entropy = log_entropy(bottom_entropy_proportion)
    top_entropy = log_entropy(top_entropy_proportion)

    # Create dictionaries
    bottom_dict = {
        "data": bottom,
        "data_proportion": bottom_weight,
        "entropy": bottom_entropy,
    }
    top_dict = {
        "data": top,
        "data_proportion": top_weight,
        "entropy": top_entropy,
    }

    return {"bottom": bottom_dict, "top": top_dict}
