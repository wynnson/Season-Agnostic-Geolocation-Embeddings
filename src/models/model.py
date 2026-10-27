import torch
from torch import nn
from torch.nn import functional as F

from src.models.gradient_reversal import gradient_reverse
from src.models.heads import build_projection_head
from src.models.patch_attention import PatchAttentionPool


class SeasonAgnosticModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        hidden_dim: int,
        projection_dropout: float,
        season_classes: int,
        projection_architecture: str = "mlp",
        patch_attention_config=None,
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
        self.patch_attention = self._build_patch_attention(patch_attention_config, backbone_dim)
        self.backbone_frozen = False

    @staticmethod
    def _build_patch_attention(
        patch_attention_config,
        backbone_dim: int,
    ) -> PatchAttentionPool | None:
        if patch_attention_config is None or not patch_attention_config.enabled:
            return None
        return PatchAttentionPool(
            embedding_dim=backbone_dim,
            hidden_dim=patch_attention_config.hidden_dim,
            residual_scale_initialization=patch_attention_config.residual_scale_initialization,
        )

    def embed_features(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        attention = None
        if self.patch_attention is not None:
            if patch_features is None:
                raise ValueError("Patch attention requires patch features")
            cls_features, attention = self.patch_attention(cls_features, patch_features)

        embedding = F.normalize(self.projection_head(cls_features), dim=-1)
        if return_attention:
            return embedding, attention
        return embedding

    def forward_feature_training(
        self,
        cls_features: torch.Tensor,
        patch_features: torch.Tensor | None,
        grl_scale: float,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        embedding, attention = self.embed_features(
            cls_features,
            patch_features,
            return_attention=True,
        )
        season_logits = self.season_classifier(gradient_reverse(embedding, grl_scale))
        if return_attention:
            return embedding, season_logits, attention
        return embedding, season_logits

    def _backbone_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.patch_attention is None:
            return self.backbone(images), None
        if not hasattr(self.backbone, "forward_features"):
            raise ValueError("Patch attention requires a backbone with forward_features")

        features = self.backbone.forward_features(images)
        try:
            return features["x_norm_clstoken"], features["x_norm_patchtokens"]
        except KeyError as error:
            raise ValueError("Backbone forward_features must return CLS and patch tokens") from error

    def forward_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        cls_features, patch_features = self._backbone_features(images)
        return self.embed_features(cls_features, patch_features)

    def forward_training(
        self, images: torch.Tensor, grl_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_features, patch_features = self._backbone_features(images)
        return self.forward_feature_training(cls_features, patch_features, grl_scale)

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
