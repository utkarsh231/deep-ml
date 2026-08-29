import torch

def cross_entropy(logits, targets):
    # logits: (N, C)
    # targets: (N,)

    shifted = logits - logits.max(dim=1, keepdim=True).values

    log_sum_exp = torch.log(
        torch.exp(shifted).sum(dim=1, keepdim=True)
    )

    log_probs = shifted - log_sum_exp

    correct_log_probs = log_probs[
        torch.arange(logits.shape[0]),
        targets
    ]

    return -correct_log_probs.mean()