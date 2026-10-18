import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using: {device}")
    return torch.device(device)
