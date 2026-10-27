import torch


def symmetric_js_divergence(
    first_attention: torch.Tensor,
    second_attention: torch.Tensor,
) -> torch.Tensor:
    if first_attention.shape != second_attention.shape:
        raise ValueError("Attention distributions must have matching shapes")

    epsilon = torch.finfo(first_attention.dtype).eps
    first_attention = first_attention.clamp_min(epsilon)
    second_attention = second_attention.clamp_min(epsilon)
    midpoint = (first_attention + second_attention) / 2
    return (
        (first_attention * (first_attention.log() - midpoint.log())).sum(dim=-1)
        + (second_attention * (second_attention.log() - midpoint.log())).sum(dim=-1)
    ).mean() / 2
