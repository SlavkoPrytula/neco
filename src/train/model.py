"""Keep NeCo's loss/ROI alignment; expose benchmark initialization and exports."""

from pathlib import Path
import sys

import torch

from src.neco import NeCo


class BenchmarkNeCo(NeCo):
    def __init__(self, cfg, fields):
        # Set before nn.Module initialization without registering a module.
        object.__setattr__(self, 'benchmark_config', cfg)
        m, t, d = cfg.model, cfg.train, cfg.data
        super().__init__(
            gpus=1, num_samples=fields, batch_size=t.batch_size, max_epochs=1,
            lr_heads=cfg.optim.lr_heads, lr_backbone=cfg.optim.lr_backbone,
            final_lr=cfg.optim.min_lr, weight_decay=cfg.optim.weight_decay,
            weight_decay_end=cfg.optim.weight_decay_end,
            projection_hidden_dim=m.projection_hidden_dim, projection_feat_dim=m.projection_dim,
            n_layers_projection_head=m.projection_layers, crops_for_assign=[0, 1],
            nmb_crops=list(d.nmb_crops), num_classes=2, val_iters=0, num_clusters_kmeans=[2],
            use_teacher=True, loss_mask='all', queue_length=m.queue_length,
            momentum_teacher=cfg.optim.momentum, momentum_teacher_end=cfg.optim.momentum_end,
            optimizer='adamw', patch_size=m.patch_size, roi_align_kernel_size=m.roi_size,
            arch='vit-base', arch_version='v2', num_register_tokens=4,
            student_steepness=m.steepness, teacher_steepness=m.steepness,
            sort_net='bitonic', trainable_blocks=None, is_queue_usable=True, sim_dist='cosine',
        )
        self.teacher.requires_grad_(False)

    def init_model(self):
        m = self.benchmark_config.model
        if m.backbone == 'dinov2r':
            # Use upstream construction verbatim, including student drop path 0.1.
            return super().init_model()
        if m.backbone == 'dinov3':
            from src.models.dinov3_backbone import DinoV3NeCo
            kwargs = dict(source_root=m.source_root, encoder_config=dict(m.encoder),
                          hidden_dim=m.projection_hidden_dim, output_dim=m.projection_dim,
                          nlayers=m.projection_layers, precision=m.precision)
            self.teacher = DinoV3NeCo(**kwargs)
            return DinoV3NeCo(**kwargs)
        raise ValueError(m.backbone)

    def initialize(self):
        m = self.benchmark_config.model
        if m.backbone == 'dinov2r':
            state = torch.load(m.pretrained, map_location='cpu', weights_only=True)
            for network in (self.model, self.teacher):
                expected = {k for k in network.state_dict() if not k.startswith('projection_head.')}
                if state.keys() != expected:
                    raise ValueError(f'Encoder key mismatch: {state.keys() ^ expected}')
                result = network.load_state_dict(state, strict=False)
                if result.unexpected_keys or any(not k.startswith('projection_head.') for k in result.missing_keys):
                    raise ValueError(result)
        else:
            sys.path.insert(0, str(Path(m.source_root).resolve()))
            from dinov3.models.adapters.io import load_pretrained_backbone_
            for network in (self.model, self.teacher):
                load_pretrained_backbone_(network.backbone, m.pretrained)

    def encoder_state(self, teacher=True):
        network = self.teacher if teacher else self.model
        if self.benchmark_config.model.backbone == 'dinov3':
            network = network.backbone
        return {k: v.detach().cpu().clone() for k, v in network.state_dict().items()
                if not k.startswith('projection_head.')}

    def runtime_state(self):
        return dict(queue=None if self.queue is None else self.queue.detach().cpu(),
                    entries_in_queue=self.entries_in_queue, use_the_queue=self.use_the_queue)

    def restore_runtime(self, state):
        self.queue = None if state['queue'] is None else state['queue'].to(self.device)
        self.entries_in_queue = state['entries_in_queue']
        self.use_the_queue = state['use_the_queue']
