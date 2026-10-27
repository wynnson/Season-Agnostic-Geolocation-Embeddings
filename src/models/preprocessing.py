import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize

from src.models.model import SeasonAgnosticModel


def preprocess_images(
    images: torch.Tensor,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    if images.shape[-2:] != (image_size, image_size):
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    normalize = Normalize(mean=mean, std=std)
    return normalize(images)


def embed_images(
    model: nn.Module,
    images: torch.Tensor,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    images = preprocess_images(images, image_size, mean, std)
    if isinstance(model, SeasonAgnosticModel):
        return model.forward_embeddings(images)
    return F.normalize(model(images), dim=1)


def extract_patch_tokens(
    model: nn.Module,
    images: torch.Tensor,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    images = preprocess_images(images, image_size, mean, std)
    backbone = model.backbone if isinstance(model, SeasonAgnosticModel) else model
    if not hasattr(backbone, "forward_features"):
        raise ValueError("Patch reranking requires a backbone with forward_features")
    features = backbone.forward_features(images)
    try:
        return features["x_norm_patchtokens"]
    except KeyError as error:
        raise ValueError("Backbone forward_features did not return x_norm_patchtokens") from error
