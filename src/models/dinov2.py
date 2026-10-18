import torch
from torch.nn import functional as F
from torchvision.transforms import Normalize


DINO_MEAN              = (0.485, 0.456, 0.406)
DINO_STD               = (0.229, 0.224, 0.225)


def load_dinov2(
    model_name: str,
    trained: bool = False,
    repository: str = "facebookresearch/dinov2",
):
    if trained:
        # TODO: Load the trained model here
        model = None
        raise NotImplementedError("Loading trained model checkpoints is not implemented yet")
    else:
        model = load_frozen_dinov2(model_name, repository)

    print(f"Using model: {model_name} (trained={trained})")
    return model


def load_frozen_dinov2(
    model_name: str = "dinov2_vits14",
    repository: str = "facebookresearch/dinov2",
):
    model = torch.hub.load(repository, model_name)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def embed_dinov2_images(model, images: torch.Tensor, image_size: int) -> torch.Tensor:
    b, c, h, w = images.shape

    if h != image_size or w != image_size:
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False
        )

    normalize = Normalize(mean=DINO_MEAN, std=DINO_STD)
    images = normalize(images)
    embeddings = model(images)
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings
