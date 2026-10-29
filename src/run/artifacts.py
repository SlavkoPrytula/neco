"""Atomic artifacts, complete recovery state and canonical benchmark provenance."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

import numpy as np
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = '393216026b4d045e2e0b803f70a7d89d9a58937e'


def wandb_metrics(metrics, batch_size):
    """Expose the coordinator's names without changing native/local loss values."""
    names = {'loss': 'total_loss', 'lr_backbone': 'lr', 'lr_heads': 'head_lr',
             'momentum': 'teacher_momentum'}
    values = {f'train/{names.get(key, key)}': value for key, value in metrics.items()}
    values.update({'train/local_batch_size': batch_size, 'train/global_batch_size': batch_size})
    return values


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def config_hash(cfg):
    return hashlib.sha256(json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temporary)
    os.replace(temporary, path)


def git_info(root):
    def run(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    return dict(source_root=str(root), commit=run('rev-parse', 'HEAD'),
                dirty=bool(run('status', '--porcelain')), repository=run('remote', 'get-url', 'origin'))


def upstream_audit():
    """Fail closed if imported loss, transforms, model or helper source changes."""
    paths = subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', UPSTREAM,
         '--', 'src', 'experiments/utils.py'], text=True).splitlines()
    hashes = {}
    for name in paths:
        original = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{UPSTREAM}:{name}'])
        current = (ROOT / name).read_bytes()
        if current != original:
            raise ValueError(f'Native NeCo source changed: {name}')
        hashes[name] = hashlib.sha256(current).hexdigest()
    return dict(revision=UPSTREAM, sha256=hashes)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_recovery(path, model, optimizer, cfg, run_id, step):
    atomic_torch(path, dict(schema_version=1, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                           runtime=model.runtime_state(), rng=rng_state(), completed_steps=step,
                           config_sha256=config_hash(cfg), run_id=run_id, world_size=1))


def load_recovery(path, model, optimizer, cfg, run_id):
    # Only our own local, recorded full-state checkpoints are accepted here.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['config_sha256'] != config_hash(cfg) or state['run_id'] != run_id or state['world_size'] != 1:
        raise ValueError('Recovery config/run/world-size mismatch')
    model.load_state_dict(state['state_dict'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    model.restore_runtime(state['runtime'])
    restore_rng(state['rng'])
    return state['completed_steps']


class RunRecord:
    def __init__(self, root, cfg, evidence, resume):
        self.root, self.cfg = Path(root), cfg
        self.path = self.root / 'run_metadata.json'
        self.started, self.start_step, self.step = time.monotonic(), 0, 0
        self.wandb = None
        self.data = json.loads(self.path.read_text()) if resume else dict(
            schema_version=1, run_id=str(uuid.uuid4()), stage='ssl', method='neco',
            model=cfg.experiment.model_name, seed=cfg.train.seed, world_size=1,
            config_sha256=config_hash(cfg), evidence=evidence, attempts=[], exports=[],
            sampler='shuffled passes seeded by seed+pass, consumed-field offset',
            resume_limitations='Worker RNG/prefetch is not replayed bitwise with num_workers>0.',
        )
        if self.data['config_sha256'] != config_hash(cfg) or self.data['evidence'] != evidence:
            raise ValueError('Saved run configuration, initial weights or manifest changed')
        sources = dict(neco={**git_info(ROOT), 'upstream_revision': UPSTREAM,
                            'native_source': upstream_audit(),
                            'integration_sha256': {str(p.relative_to(ROOT)): sha256(p)
                                for folder in ('src/train', 'src/data', 'src/run')
                                for p in sorted((ROOT / folder).rglob('*.py'))}})
        if cfg.model.backbone == 'dinov3':
            sources['dinov3'] = git_info(Path(cfg.model.source_root))
        self.attempt = dict(number=len(self.data['attempts']) + 1, started_at=now(), sources=sources,
                            python=sys.executable, packages={p: metadata.version(p) for p in
                            ('torch', 'torchvision', 'xformers', 'pytorch-lightning', 'diffsort', 'numpy', 'wandb')},
                            gpu=torch.cuda.get_device_name(), slurm={k: v for k, v in os.environ.items()
                            if k in ('SLURM_JOB_ID', 'SLURM_JOB_ACCOUNT', 'SLURM_JOB_PARTITION',
                                     'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE',
                                     'SLURM_JOB_GPUS', 'SLURM_TIMELIMIT')})
        self.data['attempts'].append(self.attempt)
        self.data['status'] = 'running'
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def setup_wandb(self):
        settings = self.cfg.logging.wandb
        if settings.mode == 'disabled':
            return
        import wandb
        directory = self.root / 'wandb'
        directory.mkdir(exist_ok=True)
        for key, child in [('WANDB_CACHE_DIR', 'cache'), ('WANDB_CONFIG_DIR', 'config'), ('WANDB_DATA_DIR', 'data')]:
            os.environ[key] = str(directory / child)
        os.environ['WANDB_DIR'] = str(self.root)
        previous = self.data.get('wandb', {})
        name = previous.get('name') or f"{self.cfg.experiment.model_name}-s{self.cfg.train.seed}-j{os.environ['SLURM_JOB_ID']}"
        self.wandb = wandb.init(project=settings.project, entity=settings.get('entity'), group=settings.group,
                               name=name, id=self.data['run_id'], job_type='ssl-posttrain',
                               mode=settings.mode, resume='allow' if settings.mode == 'online' else None,
                               dir=str(self.root), config=OmegaConf.to_container(self.cfg, resolve=True))
        if not Path(self.wandb.dir).resolve().is_relative_to(directory.resolve()):
            raise RuntimeError('W&B output escaped canonical run directory')
        self.wandb.define_metric('train/completed_steps', hidden=True)
        self.wandb.define_metric('train/*', step_metric='train/completed_steps')
        # Explicit definitions also cover metrics previously displayed with Step.
        for key in ('total_loss', 'lr', 'head_lr', 'weight_decay', 'teacher_momentum',
                    'grad_norm', 'seconds_per_update', 'local_batch_size', 'global_batch_size'):
            self.wandb.define_metric(f'train/{key}', step_metric='train/completed_steps')
        self.data['wandb'] = dict(id=self.wandb.id, url=self.wandb.url, name=self.wandb.name, mode=settings.mode)
        self.save()

    def log(self, metrics):
        with (self.root / 'metrics.json').open('a') as handle:
            handle.write(json.dumps(metrics, allow_nan=False) + '\n')
        if self.wandb:
            # Match DINOv3: ordinary Step tracks updates, while replay after a
            # recovery remains monotonic. completed_steps is always authoritative.
            self.wandb.log(wandb_metrics(metrics, self.cfg.train.batch_size),
                           step=max(metrics['completed_steps'], self.wandb.step))

    def export(self, model, final=False):
        path = self.root / ('model_final.pth' if final else f'checkpoints/ema/model_step_{self.step:08d}.pth')
        atomic_torch(path, dict(teacher=model.encoder_state(), completed_steps=self.step,
                               backbone=self.cfg.model.backbone, source_run_id=self.data['run_id'],
                               encoder_config=OmegaConf.to_container(self.cfg.model, resolve=True)))
        self.data['exports'] = [e for e in self.data['exports'] if e['path'] != str(path)] + [
            dict(path=str(path), completed_steps=self.step, sha256=sha256(path), selection='ema')]
        self.save()

    def recoverable(self, model, optimizer):
        path = self.root / 'checkpoints/resume/last.pth'
        save_recovery(path, model, optimizer, self.cfg, self.data['run_id'], self.step)
        self.data['recovery'] = dict(path=str(path), completed_steps=self.step, sha256=sha256(path))
        self.save()

    def finish(self, status, error=None):
        elapsed = time.monotonic() - self.started
        images = (self.step - self.start_step) * self.cfg.train.batch_size
        self.attempt.update(status=status, finished_at=now(), start_step=self.start_step, completed_steps=self.step,
                            wall_seconds=elapsed, gpu_hours=elapsed / 3600, images_per_second=images / elapsed,
                            image_presentations=images, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if error is not None:
            self.attempt['error'] = str(error)
        self.data.update(status=status, completed_steps=self.step,
                         image_presentations=self.step * self.cfg.train.batch_size)
        self.save()
        if self.wandb:
            self.wandb.finish(exit_code=1 if status == 'failed' else 0)
