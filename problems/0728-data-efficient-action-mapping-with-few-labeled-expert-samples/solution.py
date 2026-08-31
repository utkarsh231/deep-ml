import numpy as np

def map_latent_to_real(latent_labeled: list, real_labeled: list, latent_query: list) -> np.ndarray:
    """
    Fit an affine map from latent actions to real actions using a small labeled set,
    and apply it to a batch of latent queries.

    Args:
        latent_labeled: (N, d_latent) labeled latent actions.
        real_labeled:   (N, d_real)   corresponding real actions.
        latent_query:   (M, d_latent) latent actions to translate.

    Returns:
        numpy array of shape (M, d_real) with predicted real actions.
    """
    latent_labeled = np.array(latent_labeled, dtype=float)
    real_labeled = np.array(real_labeled, dtype=float)
    latent_query = np.array(latent_query, dtype=float)

    X = np.hstack([
        latent_labeled,
        np.ones((latent_labeled.shape[0], 1))
    ])
    params = np.linalg.lstsq(X, real_labeled, rcond = None)[0]

    X_query = np.hstack([
        latent_query,
        np.ones((latent_query.shape[0], 1))
    ])
    return X_query @ params