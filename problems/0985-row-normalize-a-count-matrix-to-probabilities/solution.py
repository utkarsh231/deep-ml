import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    arr = np.array(counts, dtype = float)

    row_sums = np.sum(arr, axis=1, keepdims = True)

    probs = np.divide(
        arr, 
        row_sums,
        out = np.zeros_like(arr), 
        where = row_sums!=0
        )

    return probs.tolist()