import torch
from torch import nn
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        hard_negative_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)

        similarity = query @ key.T
        logits = similarity / self.temperature

        if hard_negative_keys is not None:
            hard_negative_keys = F.normalize(hard_negative_keys, dim=-1)
            hard_negative_similarity = (hard_negative_keys @ query.unsqueeze(-1)).squeeze(-1)
            logits = torch.cat((logits, hard_negative_similarity / self.temperature), dim=1)

        batch_size = query.shape[0]
        targets = torch.arange(batch_size, device=query.device)
        loss = F.cross_entropy(logits, targets)
        return loss
