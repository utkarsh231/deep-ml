import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):

    # Q shape: (seq_len, d_k)

    d_k = Q.shape[1]

    # QK^T

    scores = np.dot(Q, K.T)

    # Scale

    scores = scores / np.sqrt(d_k)

    # Softmax

    scores = scores - np.max(scores, axis=1, keepdims=True)

    exp_scores = np.exp(scores)

    attention_weights = exp_scores / np.sum(

        exp_scores, axis=1, keepdims=True

    )

    # Weighted sum of V

    output = np.dot(attention_weights, V)

    return output
