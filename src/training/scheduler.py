import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    match scheduler_cfg.name:
        case "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=scheduler_cfg.min_learning_rate,
            )
        case name:
            raise ValueError(f"Unsupported scheduler: {name}")
