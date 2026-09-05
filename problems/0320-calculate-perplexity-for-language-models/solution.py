import numpy as np
import math
def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    n = len(probabilities)

    log_sum = 0

    for p in probabilities:

        log_sum += math.log(p)

    avg_log_prob = log_sum / n

    return math.exp(-avg_log_prob)
   