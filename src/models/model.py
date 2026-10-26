import torch
from torch import nn
from torch.nn import functional as F

from src.models.gradient_reversal import gradient_reverse
from src.models.heads import build_projection_head
from src.models.layer_mixer import LayerMixer


class SeasonAgnosticModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        hidden_dim: int,
        projection_dropout: float,
        season_classes: int,
        projection_architecture: str = "mlp",
        layer_mixer_config=None,
    ):
        super().__init__()
        self.backbone = backbone
        backbone_dim = getattr(backbone, "embed_dim", None)

        if not isinstance(backbone_dim, int):
            raise ValueError("Backbone must expose an integer embed_dim")

        self.projection_head = build_projection_head(
            backbone_dim,
            hidden_dim,
            embedding_dim,
            projection_dropout,
            projection_architecture,
        )
        self.season_classifier = nn.Linear(embedding_dim, season_classes)
        self.layer_mixer = self._build_layer_mixer(layer_mixer_config)
        self.backbone_frozen = False

    @staticmethod
    def _build_layer_mixer(layer_mixer_config) -> LayerMixer | None:
        if layer_mixer_config is None or not layer_mixer_config.enabled:
            return None
        return LayerMixer(list(layer_mixer_config.layers))

    def mix_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.layer_mixer is None:
            return features
        return self.layer_mixer(features)

    def _backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.layer_mixer is None:
            return self.backbone(images)
        if not hasattr(self.backbone, "get_intermediate_layers"):
            raise ValueError("Layer mixing requires a backbone with get_intermediate_layers")

        outputs = self.backbone.get_intermediate_layers(
            images,
            n=list(self.layer_mixer.layers),
            return_class_token=True,
        )
        features = torch.stack([class_token for _, class_token in outputs], dim=-2)
        return self.mix_features(features)

    def forward_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        features = self._backbone_features(images)
        embedding = self.projection_head(features)
        return F.normalize(embedding, dim=1)

    def forward_training(
        self, images: torch.Tensor, grl_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._backbone_features(images)
        embedding = self.projection_head(features)
        normalized_out = F.normalize(embedding, dim=1)
        season_features = gradient_reverse(normalized_out, grl_scale)
        season_logits = self.season_classifier(season_features)
        return normalized_out, season_logits

    def freeze_backbone(self) -> None:
        self.backbone_frozen = True
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self
