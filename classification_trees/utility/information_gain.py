from typing import List, Dict


def calculate_information_gain(
    parent_entropy: float, children: List[Dict[str, float]]
) -> float:
    """This function calculates the entropy gain when the dataframe is split.

    Parameters
    ----------
    parent_entropy : float
        The entropy of the parent node
    children : List[Dict[str, float]]
        A list of dictionaries of the form:

        ```python
        [
            {
                "data_proportion":0.5,
                "entropy": 0.5
            }
        ]
        ```
        The data proportions should always add to 1.


    Returns
    -------
    float
        The information gained from splitting the two, we aim to maximise this number as we want to minimise
        the sum of the entropies to ensure that they have sufficiently low entropy!
    """
    if sum([child["data_proportion"] for child in children]) != 1:
        raise ValueError("The data proportions do not add to one!")

    weighted_sum = [child["data_proportion"] * child["entropy"] for child in children]
    return parent_entropy - sum(weighted_sum)
