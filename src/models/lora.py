import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + self.scale * update


def apply_lora(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> int:
    if not target_modules:
        raise ValueError("LoRA target_modules must not be empty")

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    targets = set(target_modules)
    replaced = 0
    for name, module in list(backbone.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue

        leaf_name = name.rsplit(".", maxsplit=1)[-1]
        if name not in targets and leaf_name not in targets:
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = backbone if not parent_name else backbone.get_submodule(parent_name)
        parent._modules[child_name] = LoRALinear(module, rank, alpha, dropout)
        replaced += 1

    if replaced == 0:
        raise ValueError(
            f"LoRA targets {sorted(targets)} did not match any linear backbone modules"
        )

    return replaced
