from typing import List
from math import isclose

import numpy as np


def singular_log_entropy(proportion: float) -> float:
    """Calculate log entropy

    Parameters
    ----------
    proportion : float

    Returns
    -------
    float
        Log information entropy
    """
    if proportion == 0:
        return 0
    return proportion * np.log(proportion)


def log_entropy(proportions: List[float]) -> float:
    """
    Calculates log entropy for a list of proportions
    """
    # The first check we do is to see if the proportions truly sum to 1.
    if not isclose(sum(proportions), 1, abs_tol=0.03):
        raise ValueError(
            f"""
            The proportions: {proportions}
            Sum to {sum(proportions)}, not 1."""
        )

    transformed_proportions = map(singular_log_entropy, proportions)
    return -sum(transformed_proportions)
