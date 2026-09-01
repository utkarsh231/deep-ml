import numpy as np

def video_tokenizer_reconstruct(video, patch_t: int, patch_h: int, patch_w: int, mode: str):
    """
    Reconstruct a video after mean-pool tokenization.

    Args:
        video: array-like of shape (T, H, W, C)
        patch_t: temporal patch size (ignored when mode == 'spatial')
        patch_h: spatial patch height
        patch_w: spatial patch width
        mode: 'spatial' or 'spatiotemporal'

    Returns:
        Reconstructed video as a nested list of shape (T, H, W, C).
    """

    video = np.array(video, dtype = float)

    T,H,W,C = video.shape

    if mode == 'spatial':
        pt = 1
    elif mode == 'spatiotemporal':
        pt = patch_t

    reconstructed = np.zeros_like(video, dtype = float)

    for t in range(0, T, pt):
        for h in range(0, H, patch_h):
            for w in range(0, W, patch_w):

                patch = video[
                    t:t + pt,
                    h:h + patch_h,
                    w:w + patch_w,
                    :
                ]

                patch_mean = np.mean(patch, axis=(0,1,2))

                reconstructed[
                    t:t + pt,
                    h:h + patch_h,
                    w:w + patch_w,
                    :
                ] = patch_mean
    return reconstructed.tolist()

