"""Explicit Slurm-only integration test, never imported by the CPU test suite."""
import argparse
import gc
import inspect
import json
import math
import os
import random
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import default_collate

from src.run.artifacts import atomic_json, rng_state, restore_rng, sha256, load_recovery
from src.train.schedules import native_optimizer, learning_rate
from src.data.datasets.revvity import Revvity
from src.train.model import BenchmarkNeCo
from src.train.train import move_batch
from src.neco import NeCo
from src.neco_transforms import NeCoTransforms

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-native-optim-100k-bs16-lr1e-6.yaml'
MATCHED_CONFIG = ROOT / 'configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-100k-bs16-lr2e-4.yaml'


def matched_validation(root):
    """Check peak-LR execution, recovery and the real 75-step head-freeze boundary."""
    from src.train.schedules import build_optimizer
    def run(*arguments):
        subprocess.run([sys.executable, '-m', 'src.train.train', *map(str, arguments)], cwd=ROOT, check=True)
    cfg = OmegaConf.load(MATCHED_CONFIG)
    cfg.logging.wandb.mode = 'offline'
    cfg.logging.interval = 1
    cfg.checkpointing.period = cfg.checkpointing.export_period = 4
    # Explicit compressed diagnostic exercises the peak LR; not a research recipe.
    short = OmegaConf.create(OmegaConf.to_container(cfg))
    short.train.max_steps = 8
    short.optim.warmup_steps = 2
    short.optim.freeze_last_layer_steps = 1
    short.train.num_workers = 0
    fixture = root / 'matched-short.yaml'
    OmegaConf.save(short, fixture)
    output = root / 'matched-short'
    run('--config-file', fixture, '--output-dir', output, '--stop-after', 4)
    run('--run-dir', output)
    state = torch.load(output / 'checkpoints/resume/last.pth', map_location='cpu', weights_only=False)
    model = BenchmarkNeCo(short, 8938).cuda().train()
    model.on_train_epoch_start()
    optimizer = build_optimizer(model, short)
    assert load_recovery(output / 'checkpoints/resume/last.pth', model, optimizer, short, state['run_id']) == 8
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), state['state_dict'][name], rtol=0, atol=0)
    saved_optimizer = state['optimizer']['state']
    for index, values in optimizer.state_dict()['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(value.cpu(), saved_optimizer[index][key], rtol=0, atol=0)
    export = torch.load(output / 'model_final.pth', map_location='cpu', weights_only=True)
    for name, tensor in export['teacher'].items():
        torch.testing.assert_close(tensor, state['state_dict']['teacher.' + name], rtol=0, atol=0)
    short_metrics = [json.loads(s) for s in (output / 'metrics.json').read_text().splitlines()]
    assert max(m['lr_backbone'] for m in short_metrics) == 2e-4
    assert all(math.isfinite(m['loss']) and math.isfinite(m['grad_norm']) for m in short_metrics)
    del model, optimizer, state, export
    gc.collect(); torch.cuda.empty_cache()
    # Actual 100k horizon, 750-step warmup, 75-step final-layer freeze, 8 workers.
    cfg.checkpointing.period = cfg.checkpointing.export_period = 16
    fixture = root / 'matched-production.yaml'
    OmegaConf.save(cfg, fixture)
    output = root / 'matched-production'
    run('--config-file', fixture, '--output-dir', output, '--stop-after', 74)
    state = torch.load(output / 'checkpoints/resume/last.pth', map_location='cpu', weights_only=False)
    final_layer = 'model.projection_head.4.weight'
    before = state['state_dict'][final_layer].clone()
    del state
    # Resume CLI runs to max_steps, so stop-after is explicitly passed for this diagnostic.
    run('--run-dir', output, '--stop-after', 80)
    state = torch.load(output / 'checkpoints/resume/last.pth', map_location='cpu', weights_only=False)
    assert state['completed_steps'] == 80 and not torch.equal(before, state['state_dict'][final_layer])
    assert state['runtime']['entries_in_queue'] == cfg.model.queue_length
    metrics = [json.loads(s) for s in (output / 'metrics.json').read_text().splitlines()]
    assert [m['completed_steps'] for m in metrics] == list(range(1, 81))
    for m in metrics:
        assert math.isfinite(m['loss']) and math.isfinite(m['grad_norm'])
        assert math.isclose(m['lr_backbone'], learning_rate(2e-4, 1e-6, m['completed_steps'] - 1, 100000, 750), abs_tol=1e-15)
    metadata = json.loads((output / 'run_metadata.json').read_text())
    return dict(status='passed', peak_lr_test_steps=8, production_steps=80,
                exact_model_optimizer_reload=True, final_export_matches_teacher=True,
                final_projection_unfrozen_after_step75=True, queue_saturated=True,
                first_loss=metrics[0]['loss'], last_loss=metrics[-1]['loss'],
                peak_allocated_bytes=max(a['peak_allocated_bytes'] for a in metadata['attempts']))


def parity(cfg):
    """Compare native and wrapped forward, loss, student gradients, queue and EMA."""
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    wrapped = BenchmarkNeCo(cfg, 8938).cuda().train()
    wrapped.initialize()
    kwargs = {k: v for k, v in wrapped.hparams.items() if k in inspect.signature(NeCo).parameters}
    native = NeCo(**kwargs).cuda().train()
    native.load_state_dict(wrapped.state_dict(), strict=True)
    for model in (wrapped, native):
        model.on_train_epoch_start()
    assert wrapped.model.num_register_tokens == native.model.num_register_tokens == 4
    assert math.isclose(wrapped.model.blocks[-1].drop_path1.drop_prob, .1, abs_tol=1e-8)
    # Real microscopy and native geometry. Small batch only for the duplicate-model parity check.
    d = cfg.data
    transform = NeCoTransforms(list(d.size_crops), list(d.nmb_crops), list(d.min_scale_crops),
                               list(d.max_scale_crops), d.jitter_strength, d.min_intersection, d.blur_strength)
    dataset = Revvity(d.manifest, transform)
    batch = move_batch(default_collate([dataset[0], dataset[1]]))
    state = rng_state()
    def forward_backward(model):
        model.zero_grad(set_to_none=True)
        model.queue = None
        model.entries_in_queue = 0
        model.use_the_queue = False
        model.on_train_epoch_start()
        restore_rng(state)
        loss = model.shared_step(batch)
        loss.backward()
        return loss

    expected = forward_backward(native)
    reference = {name: p.grad.detach().clone() for name, p in native.model.named_parameters()
                 if p.grad is not None}
    repeated = forward_backward(native)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    actual = forward_backward(wrapped)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    count = 0
    differences = []
    for (name, a), (other, b) in zip(wrapped.model.named_parameters(), native.model.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            # Native CUDA backward is not bitwise deterministic (including bicubic
            # position interpolation). Native-vs-native controls measured 2–5e-6
            # absolute differences. Retain a strict relative-norm check as well.
            row = {'name': name}
            for label, gradient in [('native_repeat', b.grad), ('wrapper', a.grad)]:
                target = reference[name]
                torch.testing.assert_close(gradient, target, rtol=1e-4, atol=1e-5, msg=f'{label}: {name}')
                relative = float(torch.linalg.vector_norm(gradient - target) /
                                 torch.linalg.vector_norm(target).clamp_min(1e-12))
                assert relative < 1e-4, (label, name, relative)
                row[label] = dict(max_absolute=float((gradient - target).abs().max()), relative_l2=relative)
            differences.append(row)
            assert torch.isfinite(a.grad).all(), name
            count += 1
    torch.testing.assert_close(wrapped.queue, native.queue, rtol=0, atol=0)
    assert wrapped.entries_in_queue == native.entries_in_queue
    # Upstream uses .data EMA; wrapper uses frozen teacher under no_grad.
    with torch.no_grad():
        for s, t in zip(wrapped.model.parameters(), wrapped.teacher.parameters()):
            t.mul_(cfg.optim.momentum).add_(s, alpha=1 - cfg.optim.momentum)
        for s, t in zip(native.model.parameters(), native.teacher.parameters()):
            t.data.mul_(cfg.optim.momentum).add_((1 - cfg.optim.momentum) * s.detach().data)
        for a, b in zip(wrapped.teacher.parameters(), native.teacher.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    value = dict(loss=float(actual.detach()), student_gradient_tensors=count,
                 encoder_parameters=sum(p.numel() for n, p in wrapped.model.named_parameters()
                                        if not n.startswith('projection_head.')),
                 head_parameters=sum(p.numel() for n, p in wrapped.model.named_parameters()
                                     if n.startswith('projection_head.')),
                 registers=4, native_loss_and_gradients_match=True)
    value['gradient_comparison'] = dict(rtol=1e-4, atol=1e-5, relative_l2_limit=1e-4,
                                        per_parameter=differences)
    del wrapped, native, batch, actual, expected, repeated, reference
    gc.collect()
    torch.cuda.empty_cache()
    return value


def compare_runs(resumed, continuous, cfg):
    a = torch.load(resumed / 'checkpoints/resume/last.pth', map_location='cpu', weights_only=False)
    b = torch.load(continuous / 'checkpoints/resume/last.pth', map_location='cpu', weights_only=False)
    assert a['completed_steps'] == b['completed_steps'] == 4
    maximum = 0.
    for key, tensor in a['state_dict'].items():
        other = b['state_dict'][key]
        # Measured CUDA trajectories differ by about 1e-6 in head weights.
        torch.testing.assert_close(tensor, other, rtol=1e-5, atol=2e-6, msg=key)
        if tensor.numel():
            maximum = max(maximum, float((tensor - other).abs().max()))
    torch.testing.assert_close(a['runtime']['queue'], b['runtime']['queue'], rtol=1e-5, atol=2e-6)
    assert a['runtime']['entries_in_queue'] == b['runtime']['entries_in_queue']
    for index, state in a['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b['optimizer']['state'][index][key], rtol=1e-4, atol=2e-7)
    metadata = json.loads((resumed / 'run_metadata.json').read_text())
    assert metadata['status'] == 'complete' and len(metadata['attempts']) == 2
    assert metadata['run_id'] == a['run_id'] == metadata['wandb']['id']
    assert 'neco-dinov2r' in metadata['wandb']['name']
    assert sha256(metadata['recovery']['path']) == metadata['recovery']['sha256']
    export = torch.load(resumed / 'model_final.pth', map_location='cpu', weights_only=True)
    raw = torch.load(cfg.model.pretrained, map_location='cpu', weights_only=True)
    assert export['teacher'].keys() == raw.keys()
    for key, value in export['teacher'].items():
        torch.testing.assert_close(value, a['state_dict']['teacher.' + key], rtol=0, atol=0)
    # Every transformer block must receive an update, including the earliest block.
    for block in range(12):
        key = f'blocks.{block}.attn.qkv.weight'
        assert not torch.equal(a['state_dict']['model.' + key], raw[key]), key
    metrics = [json.loads(line) for line in (resumed / 'metrics.json').read_text().splitlines()]
    assert [m['completed_steps'] for m in metrics] == [1, 2, 3, 4]
    continuous_metrics = [json.loads(line) for line in (continuous / 'metrics.json').read_text().splitlines()]
    torch.testing.assert_close(torch.tensor([m['loss'] for m in metrics]),
                               torch.tensor([m['loss'] for m in continuous_metrics]), rtol=1e-5, atol=2e-6)
    # Recovery must copy saved values exactly before any nondeterministic CUDA work.
    fresh = BenchmarkNeCo(cfg, 8938).cuda().train()
    fresh.on_train_epoch_start()
    optimizer = native_optimizer(fresh)
    load_recovery(resumed / 'checkpoints/resume/last.pth', fresh, optimizer, cfg, a['run_id'])
    for key, value in fresh.state_dict().items():
        torch.testing.assert_close(value.cpu(), a['state_dict'][key], rtol=0, atol=0)
    torch.testing.assert_close(fresh.queue.cpu(), a['runtime']['queue'], rtol=0, atol=0)
    assert fresh.entries_in_queue == a['runtime']['entries_in_queue']
    del fresh, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return dict(resume_max_parameter_difference=maximum, completed_steps=4,
                final_export_matches_teacher=True, all_12_blocks_updated=True,
                exact_state_reload=True, run_id=metadata['run_id'], wandb_name=metadata['wandb']['name'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU validation requires a Slurm allocation')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(CONFIG)
    result = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'status': 'running'}
    atomic_json(root / 'validation.json', result)
    try:
        result['parity'] = parity(cfg)
        atomic_json(root / 'validation.json', result)
        cfg.train.max_steps = 4
        cfg.train.num_workers = 0  # Exact RNG replay check; production uses workers.
        cfg.checkpointing.period = cfg.checkpointing.export_period = 2
        cfg.logging.interval = 1
        cfg.logging.wandb.mode = 'offline'
        fixture = root / 'diagnostic.yaml'
        OmegaConf.save(cfg, fixture)
        def run(*arguments):
            subprocess.run([sys.executable, '-m', 'src.train.train', *map(str, arguments)],
                           cwd=ROOT, check=True)
        resumed, continuous = root / 'resumed', root / 'continuous'
        run('--config-file', fixture, '--output-dir', resumed, '--stop-after', 2)
        run('--run-dir', resumed)
        run('--config-file', fixture, '--output-dir', continuous)
        result.update(compare_runs(resumed, continuous, cfg))
        # Independent uninterrupted control bounds natural CUDA trajectory noise.
        repeat = root / 'continuous-repeat'
        run('--config-file', fixture, '--output-dir', repeat)
        result['repeat_control'] = compare_runs(resumed, repeat, cfg)
        # Exercise real worker count, uncompressed LR/EMA horizon and queue saturation.
        production_cfg = OmegaConf.load(CONFIG)
        production_cfg.logging.wandb.mode = 'offline'
        production_cfg.logging.interval = 1
        production_cfg.checkpointing.period = production_cfg.checkpointing.export_period = 16
        production_fixture = root / 'production-diagnostic.yaml'
        OmegaConf.save(production_cfg, production_fixture)
        production = root / 'production'
        run('--config-file', production_fixture, '--output-dir', production, '--stop-after', 32)
        metadata = json.loads((production / 'run_metadata.json').read_text())
        metrics = [json.loads(line) for line in (production / 'metrics.json').read_text().splitlines()]
        state = torch.load(production / 'checkpoints/resume/last.pth', map_location='cpu', weights_only=False)
        assert metadata['status'] == 'stopped' and metadata['completed_steps'] == 32
        assert state['runtime']['entries_in_queue'] == production_cfg.model.queue_length
        assert [m['completed_steps'] for m in metrics] == list(range(1, 33))
        for m in metrics:
            assert math.isfinite(m['loss']) and math.isfinite(m['grad_norm']) and m['grad_norm'] > 0
            assert math.isclose(m['lr_backbone'], learning_rate(1e-6, 0, m['completed_steps'] - 1, 100000, 0), rel_tol=1e-12)
        assert len(metadata['exports']) == 2
        result['production'] = dict(completed_steps=32, schedule_steps=100000,
                                    num_workers=production_cfg.train.num_workers, queue_saturated=True,
                                    first_loss=metrics[0]['loss'], last_loss=metrics[-1]['loss'],
                                    resources=metadata['attempts'][-1])
        result['matched_optimization'] = matched_validation(root)
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        atomic_json(root / 'validation.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('parity', 'production')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
