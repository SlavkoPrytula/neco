"""Native sibling DINOv3 encoder with NeCo's projection/patch interface."""

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


class DinoV3NeCo(nn.Module):
    def __init__(self, source_root, encoder_config, hidden_dim=2048, output_dim=256, nlayers=3, precision='fp32'):
        super().__init__()
        source_root = Path(source_root).resolve()
        if not (source_root / 'dinov3/models/vision_transformer.py').is_file():
            raise FileNotFoundError(source_root)
        sys.path.insert(0, str(source_root))
        from dinov3.models.vision_transformer import vit_base
        self.backbone = vit_base(**encoder_config)
        self.backbone.init_weights()
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        self.precision = precision
        layers = [nn.Linear(self.embed_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.projection_head = nn.Sequential(*layers)

    def forward_backbone(self, x, last_self_attention=False):
        if last_self_attention:
            raise ValueError('The DINOv3 adaptation supports NeCo loss_mask=all only')
        features = self.backbone.forward_features(x)
        return torch.cat([features['x_norm_clstoken'].unsqueeze(1), features['x_norm_patchtokens']], dim=1)

    def forward(self, inputs, last_self_attention=False):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
            return self._forward_crops(inputs, last_self_attention)

    def _forward_crops(self, inputs, last_self_attention=False):
        inputs = inputs if isinstance(inputs, list) else [inputs]
        outputs = []
        start = 0
        while start < len(inputs):
            end = start + 1
            while end < len(inputs) and inputs[end].shape[-2:] == inputs[start].shape[-2:]:
                end += 1
            features = self.forward_backbone(torch.cat(inputs[start:end]), last_self_attention)
            outputs.append(features[:, 1:].reshape(-1, self.embed_dim))
            start = end
        projected = self.projection_head(torch.cat(outputs))
        # Sorting/ROI alignment stay in FP32, independent of encoder/head compute.
        return (F.normalize(projected, dim=-1, p=2).float(),)
