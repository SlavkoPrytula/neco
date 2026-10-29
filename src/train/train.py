"""Single-GPU benchmark orchestration using upstream NeCo shared_step unchanged."""

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.run.artifacts import RunRecord, load_recovery, sha256, upstream_audit
from src.data.loaders import make_dataset, validate_data
from src.data.samplers import FieldSampler, seed_worker
from .schedules import build_optimizer, set_schedule, clip_gradients, update_teacher


def validate(cfg, check_files=True):
    m, d, t, o = cfg.model, cfg.data, cfg.train, cfg.optim
    if m.backbone != 'dinov2r' or m.registers != 4:
        raise ValueError('Only native DINOv2R with four registers is validated; DINOv3 remains an inactive draft')
    if m.patch_size != (14 if m.backbone == 'dinov2r' else 16):
        raise ValueError('Wrong backbone patch size')
    if m.precision != 'fp32':
        raise ValueError('The native DINOv2R benchmark requires FP32')
    if len(d.size_crops) != 2 or list(d.nmb_crops)[0] != 2 or d.nmb_crops[1] <= 0:
        raise ValueError('NeCo requires two global and positive local crops')
    if any(s <= 0 or s % m.patch_size for s in d.size_crops):
        raise ValueError('Crop sizes must be positive patch-size multiples')
    if m.projection_layers < 2 or m.queue_length < t.batch_size * m.roi_size ** 2:
        raise ValueError('Projection/queue dimensions cannot serve this batch')
    if not 0 < d.min_intersection < 1:
        raise ValueError('Invalid overlap constraint')
    if len(d.min_scale_crops) != 2 or len(d.max_scale_crops) != 2:
        raise ValueError('Expected two crop-area ranges')
    if any(not 0 < low <= high <= 1 for low, high in zip(d.min_scale_crops, d.max_scale_crops)):
        raise ValueError('Invalid crop area range')
    if not (t.max_steps > 0 and t.batch_size > 0 and t.num_workers >= 0 and 0 <= o.warmup_steps < t.max_steps):
        raise ValueError('Invalid training budget')
    if min(cfg.checkpointing.period, cfg.checkpointing.export_period, cfg.logging.interval) <= 0:
        raise ValueError('Checkpoint/log intervals must be positive')
    if not 0 <= o.momentum <= o.momentum_end <= 1 or min(o.lr_backbone, o.lr_heads) < o.min_lr:
        raise ValueError('Invalid learning-rate/EMA endpoints')
    if o.min_lr < 0 or min(o.weight_decay, o.weight_decay_end) < 0:
        raise ValueError('Learning rate and decay must be nonnegative')
    policy = o.get('policy', 'native')
    if policy not in ('native', 'dinov3'):
        raise ValueError('Unknown optimization policy')
    if tuple(o.betas) != (0.9, 0.999):
        raise ValueError('Both prepared policies use AdamW betas (0.9, 0.999)')
    if policy == 'native' and o.clip_grad is not None:
        raise ValueError('Native NeCo uses no gradient clipping')
    if policy == 'dinov3':
        if o.clip_grad is None or o.clip_grad <= 0 or not 0 < o.layerwise_decay <= 1 or o.patch_embed_lr_mult <= 0:
            raise ValueError('Invalid DINOv3 clipping/layer-decay settings')
        if not 0 <= o.freeze_last_layer_steps < t.max_steps:
            raise ValueError('Invalid final projection-layer freeze duration')
        if o.ema_update != 'after_optimizer':
            raise ValueError('DINOv3 policy updates EMA after the optimizer')
    if cfg.logging.wandb.mode not in ('online', 'offline', 'disabled'):
        raise ValueError('Invalid W&B mode')
    if m.backbone == 'dinov3' and not (Path(m.source_root) / 'dinov3/models/vision_transformer.py').is_file():
        raise FileNotFoundError(m.source_root)
    upstream_audit()
    manifest = validate_data(d, check_files)
    if sha256(m.pretrained) != m.pretrained_sha256:
        raise ValueError('Pretrained checkpoint hash mismatch')
    return dict(manifest=manifest,
                initial_checkpoint=dict(path=m.pretrained, sha256=m.pretrained_sha256),
                max_steps=t.max_steps, batch_size=t.batch_size, image_presentations=t.max_steps * t.batch_size,
                precision=m.precision, crop_sizes=list(d.size_crops), crop_counts=list(d.nmb_crops))


def move_batch(batch):
    crops, boxes = batch
    return [x.cuda(non_blocking=True) for x in crops], {k: v.cuda(non_blocking=True) for k, v in boxes.items()}


def execute(cfg, output, evidence, resume, stop_after):
    from src.train.model import BenchmarkNeCo
    from src.neco_transforms import NeCoTransforms
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    record = RunRecord(output, cfg, evidence, resume)
    try:
        record.setup_wandb()
        model = BenchmarkNeCo(cfg, evidence['manifest']['fields']).cuda().train()
        model.initialize()
        # Same initialization as upstream: pretrained encoders, independently initialized heads.
        model.on_train_epoch_start()
        optimizer = build_optimizer(model, cfg)
        if tuple(cfg.optim.betas) != optimizer.defaults['betas']:
            raise ValueError('Keep native NeCo AdamW betas (0.9, 0.999)')
        if resume:
            recovery = record.data['recovery']
            if sha256(recovery['path']) != recovery['sha256']:
                raise ValueError('Recovery artifact hash mismatch')
            record.step = load_recovery(recovery['path'], model, optimizer, cfg, record.data['run_id'])
            if record.step != recovery['completed_steps']:
                raise ValueError('Recovery step disagrees with metadata')
        record.start_step = record.step
        if not record.step < cfg.train.max_steps or (stop_after is not None and not record.step < stop_after <= cfg.train.max_steps):
            raise ValueError('Requested stop must exceed recovery step and fit within schedule')
        end = stop_after or cfg.train.max_steps
        d = cfg.data
        transforms = NeCoTransforms(size_crops=list(d.size_crops), nmb_crops=list(d.nmb_crops),
                                    min_scale_crops=list(d.min_scale_crops), max_scale_crops=list(d.max_scale_crops),
                                    min_intersection=d.min_intersection,
                                    jitter_strength=d.jitter_strength, blur_strength=d.blur_strength)
        dataset = make_dataset(d, transforms)
        sampler = FieldSampler(len(dataset), cfg.train.seed, record.step * cfg.train.batch_size,
                               (end - record.step) * cfg.train.batch_size)
        loader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler,
                            num_workers=cfg.train.num_workers, drop_last=True, pin_memory=True,
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(cfg.train.seed))
        logger = logging.getLogger('neco.benchmark')
        logger.info('START run=%s step=%d total=%d encoder=%s precision=%s',
                    record.data['run_id'], record.step, cfg.train.max_steps, cfg.model.backbone, cfg.model.precision)
        for batch in loader:
            start = time.monotonic()
            step = record.step
            schedule = set_schedule(optimizer, cfg, step)
            optimizer.zero_grad(set_to_none=True)
            loss = model.shared_step(move_batch(batch))
            if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss).all():
                raise FloatingPointError(f'Non-finite NeCo loss at {step}: {loss}')
            if cfg.optim.get('ema_update', 'before_optimizer') == 'before_optimizer':
                update_teacher(model, schedule['momentum'])
            loss.backward()
            norm = clip_gradients(model, cfg)
            optimizer.step()
            if cfg.optim.get('ema_update', 'before_optimizer') == 'after_optimizer':
                update_teacher(model, schedule['momentum'])
            record.step = step + 1
            if record.step % cfg.logging.interval == 0 or record.step == end:
                torch.cuda.synchronize()
                metrics = dict(completed_steps=record.step, loss=float(loss.detach()), grad_norm=float(norm),
                               seconds_per_update=time.monotonic() - start, attempt=record.attempt['number'], **schedule)
                record.log(metrics)
                logger.info('Training [%d/%d] loss=%.6f lr=%.8g grad=%.4f time=%.3f',
                            record.step, cfg.train.max_steps, metrics['loss'], schedule['lr_backbone'],
                            metrics['grad_norm'], metrics['seconds_per_update'])
            del loss, batch
            if record.step % cfg.checkpointing.export_period == 0:
                record.export(model)
            if record.step % cfg.checkpointing.period == 0 or record.step == end:
                record.recoverable(model, optimizer)
        if record.step != end:
            raise RuntimeError('Data stream ended before the requested budget')
        if end == cfg.train.max_steps:
            record.export(model, final=True)
        record.finish('complete' if end == cfg.train.max_steps else 'stopped')
    except BaseException as error:
        record.finish('failed', error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config-file')
    source.add_argument('--run-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('opts', nargs='*')
    args = parser.parse_args()
    resume = args.run_dir is not None
    if resume and (args.output_dir or args.opts):
        parser.error('Recovery uses the saved config; overrides are forbidden')
    cfg = OmegaConf.load(Path(args.run_dir) / 'config.yaml' if resume else args.config_file)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))
    evidence = validate(cfg)
    if args.check_only:
        print(json.dumps(evidence, indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Training must run in a GPU Slurm allocation')
    if int(os.environ.get('SLURM_NTASKS', '1')) != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Only single-GPU training/recovery is supported')
    if cfg.model.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 requires compatible GPU hardware')
    if not (args.run_dir or args.output_dir):
        parser.error('A fresh run requires --output-dir')
    output = Path(args.run_dir or args.output_dir).resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, output / 'config.yaml')
    with (output / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (output / 'logs').mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(output / 'logs/log.txt')])
        execute(cfg, output, evidence, resume, args.stop_after)


if __name__ == '__main__':
    main()
