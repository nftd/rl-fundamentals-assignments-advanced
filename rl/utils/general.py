# general.py

import os
import numpy as np

# Set a global random seed for reproducibility
np.random.seed(42)


def set_filepath(filepath: str) -> str:
    """
    Set the filepath. If any directories do not currently exist in the filepath
    (which may be nested, e.g., /a/b/c/), create them.

    Args:
        filepath (str): The path to set.

    Returns:
        str: The validated filepath.
    """
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    return filepath


def argmax_ties_random(q_values: np.ndarray)-> int:
    """
    Gets the argmax of the q-values. Splits ties randomly.

    Args:
        q_values (np.ndarray): The q-values for each action.

    Returns:
        int: The action to take.
    """
    for index in range(len(q_values)):
        if q_values[index] == np.max(q_values):
            return np.random.choice(np.where(q_values == q_values[index])[0])



def argmax_ties_last(q_values: np.ndarray) -> int:
    """
    Gets the argmax of the q-values. Splits ties by selecting the last tied action.

    Args:
        q_values (np.ndarray): The q-values for each action.

    Returns:
        int: The action to take.
    """
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = [i]

        elif q_values[i] == top:
            ties.append(i)

    return ties[-1]
