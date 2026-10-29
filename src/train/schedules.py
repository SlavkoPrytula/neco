"""Update-addressed schedules and parameter groups for controlled continuation."""

import math
import torch


def cosine(start, end, step, total):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(step, total) / total))


def learning_rate(peak, minimum, step, total, warmup):
    if warmup and step < warmup:
        # Match the baseline's np.linspace(0, peak, warmup) warmup.
        return peak * step / max(1, warmup - 1)
    return cosine(peak, minimum, step - warmup, total - warmup)


def native_optimizer(model):
    # Preserve upstream parameter grouping, exclusions and AdamW defaults.
    optimizers, _ = model.configure_optimizers()
    optimizer = optimizers[0]
    for index, group in enumerate(optimizer.param_groups):
        group.update(is_head=index >= 2, wd_multiplier=float(index in (0, 2)))
    return optimizer


def parameter_options(name, depth, cfg, last_projection):
    """Map the DINOv3 optimization policy onto NeCo's parameter names."""
    is_head = name.startswith('projection_head.')
    layer = depth + 1
    if not is_head:
        if any(token in name for token in ('pos_embed', 'patch_embed', 'mask_token', 'cls_token', 'register_tokens')):
            layer = 0
        elif name.startswith('blocks.'):
            layer = int(name.split('.')[1]) + 1
    multiplier = cfg.optim.layerwise_decay ** (depth + 1 - layer)
    if 'patch_embed' in name:
        multiplier *= cfg.optim.patch_embed_lr_mult
    no_decay = name.endswith('bias') or any(token in name for token in ('norm', 'gamma', 'fourier_w'))
    return dict(is_head=is_head, lr_multiplier=multiplier,
                wd_multiplier=float(not no_decay), is_last_layer=name.startswith(last_projection + '.'))


def build_optimizer(model, cfg):
    if cfg.optim.get('policy', 'native') == 'native':
        return native_optimizer(model)
    head = model.model.projection_head
    linear_names = [name for name, layer in head.named_modules() if isinstance(layer, torch.nn.Linear)]
    if not linear_names:
        raise ValueError('Expected a native projection head with linear layers')
    last_projection = 'projection_head' + ('.' + linear_names[-1] if linear_names[-1] else '')
    groups = []
    for name, parameter in model.model.named_parameters():
        if parameter.requires_grad:
            groups.append(dict(params=[parameter], name=name,
                               **parameter_options(name, len(model.model.blocks), cfg, last_projection)))
    return torch.optim.AdamW(groups, betas=tuple(cfg.optim.betas), eps=1e-8)


def clip_gradients(model, cfg):
    parameters = [(name, p) for name, p in model.model.named_parameters() if p.requires_grad]
    if cfg.optim.get('policy', 'native') == 'native':
        return torch.nn.utils.clip_grad_norm_([p for _, p in parameters], float('inf'), error_if_nonfinite=True)
    # DINOv3 clips backbone and heads separately; NeCo has one projection head.
    norms = [torch.nn.utils.clip_grad_norm_(
        [p for name, p in parameters if name.startswith('projection_head.') == is_head],
        cfg.optim.clip_grad, error_if_nonfinite=True) for is_head in (False, True)]
    return torch.linalg.vector_norm(torch.stack(norms))


@torch.no_grad()
def update_teacher(model, momentum):
    for student, teacher in zip(model.model.parameters(), model.teacher.parameters()):
        teacher.mul_(momentum).add_(student, alpha=1 - momentum)


def set_schedule(optimizer, cfg, step):
    o, total = cfg.optim, cfg.train.max_steps
    backbone_lr = learning_rate(o.lr_backbone, o.min_lr, step, total, o.warmup_steps)
    head_lr = learning_rate(o.lr_heads, o.min_lr, step, total, o.warmup_steps)
    wd = cosine(o.weight_decay, o.weight_decay_end, step, total)
    momentum = cosine(o.momentum, o.momentum_end, step, total)
    for group in optimizer.param_groups:
        group['lr'] = (head_lr if group['is_head'] else backbone_lr) * group.get('lr_multiplier', 1.0)
        if group.get('is_last_layer', False) and step < o.get('freeze_last_layer_steps', 0):
            group['lr'] = 0.0
        group['weight_decay'] = wd * group['wd_multiplier']
    return dict(lr_backbone=backbone_lr, lr_heads=head_lr, weight_decay=wd, momentum=momentum)
