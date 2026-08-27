import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    if method == 'tril':
        mask = np.tril(np.ones_like(attn_weights))

        masked = attn_weights * mask

        result = masked / masked.sum(axis = 1, keepdims = True)
    
    elif method == 'triu':
        scores = np.log(attn_weights)
        mask = np.triu( np.ones_like(attn_weights, dtype = bool), k = 1)
        scores[mask] = -np.inf

        exp_scores = np.exp(scores)

        result = exp_scores / exp_scores.sum(axis = 1, keepdims = True)
    
    else:
        raise ValueError("error")

    return result.tolist()
    #pass
