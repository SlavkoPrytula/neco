"""CPU-only contract checks; full encoders are tested exclusively under Slurm."""
import csv
import importlib.util
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch

from src.run.artifacts import save_recovery, load_recovery, upstream_audit
from src.data.datasets.revvity import Revvity
from src.data.samplers import FieldSampler
from src.train.schedules import learning_rate, cosine, native_optimizer, set_schedule
from src.train.model import BenchmarkNeCo
from src.train.train import validate
from src.neco import NeCo
from experiments.utils import cosine_scheduler

ROOT = Path(__file__).resolve().parents[1]


def test_native_source_and_methods_untouched():
    assert 'src/neco.py' in upstream_audit()['sha256']
    for method in ('shared_step', 'knn_loss', 'compute_knn_perm_mat', 'compute_knn_subloss',
                   'update_queue', 'configure_optimizers', 'on_train_epoch_start'):
        assert getattr(BenchmarkNeCo, method) is getattr(NeCo, method)


def test_manifest_and_pixels_match_existing_loader(tmp_path):
    high = np.array([[0, 1, 256], [1000, 2000, 4095]], dtype=np.uint16)
    low = np.array([[4095, 3, 400], [3000, 4000, 60000]], dtype=np.uint16)
    for name, values in [('high', high), ('low', low)]:
        Image.fromarray(values).save(tmp_path / f'{name}.tif')
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sample_id', 'bf_higher', 'bf_lower'])
        writer.writeheader()
        writer.writerow(dict(sample_id='one', bf_higher=tmp_path / 'high.tif', bf_lower=tmp_path / 'low.tif'))
    spec = importlib.util.spec_from_file_location('baseline_revvity', ROOT.parent / 'dinov2/dinov2/data/datasets/revvity.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    np.testing.assert_array_equal(np.array(Revvity(manifest)[0]), np.array(baseline.Revvity(tmp_path).get_image(0)))
    with manifest.open('a') as handle:
        handle.write(f'one,{tmp_path}/high.tif,{tmp_path}/low.tif\n')
    with pytest.raises(ValueError, match='unique'):
        Revvity(manifest)


def test_sampler_resume_across_passes():
    full = list(FieldSampler(11, 42, 0, 100))
    assert list(FieldSampler(11, 42, 37, 63)) == full[37:]
    for start in range(0, 88, 11):
        assert sorted(full[start:start + 11]) == list(range(11))
    assert full != list(FieldSampler(11, 0, 0, 100))


def test_native_schedule_and_parameter_groups():
    # Native grouping tested on a tiny encoder/head, without constructing a ViT.
    class Tiny:
        model = torch.nn.ModuleDict({'encoder': torch.nn.Linear(2, 2),
                                     'projection_head': torch.nn.Linear(2, 2)})
        trainable_blocks = None
        exclude_norm_bias = True
        exclude_from_wt_decay = staticmethod(NeCo.exclude_from_wt_decay)
        weight_decay, lr_backbone, lr_heads = .04, 1e-6, 1e-4
        optim, train_iters_per_epoch, max_epochs, final_lr = 'adamw', 100, 1, 0
        configure_optimizers = NeCo.configure_optimizers
    optimizer = native_optimizer(Tiny())
    cfg = OmegaConf.load(ROOT / 'configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-native-optim-100k-bs16-lr1e-6.yaml')
    set_schedule(optimizer, cfg, 0)
    assert [g['lr'] for g in optimizer.param_groups] == [1e-6, 1e-6, 1e-4, 1e-4]
    assert [g['weight_decay'] for g in optimizer.param_groups] == pytest.approx([.04, 0, .04, 0])
    set_schedule(optimizer, cfg, 50000)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(5e-7)
    assert cosine(.9995, 1, 100000, 100000) == 1
    assert learning_rate(1e-4, 0, 100000, 100000, 0) == 0


def test_schedules_match_upstream_over_full_horizon():
    total = 100000
    for start, end in [(.04, .5), (.9995, 1.)]:
        native = cosine_scheduler(start, end, 1, total)
        for step in (0, 1, 50000, 99999):
            assert cosine(start, end, step, total) == pytest.approx(native[step], abs=1e-15)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    for step in range(20):
        assert learning_rate(1e-6, 0, step, total, 0) == pytest.approx(optimizer.param_groups[0]['lr'], rel=1e-12)
        optimizer.step()
        scheduler.step()


def test_recovery_restores_optimizer_queue_and_rng(tmp_path):
    class Tiny(torch.nn.Linear):
        def runtime_state(self):
            return self.runtime
        def restore_runtime(self, state):
            self.runtime = state
    model = Tiny(2, 1)
    model.runtime = {'queue': torch.arange(8), 'entries_in_queue': 8}
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    cfg = OmegaConf.create({'test': 1})
    path = tmp_path / 'last.pth'
    save_recovery(path, model, opt, cfg, 'uuid', 17)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    fresh = Tiny(2, 1)
    fresh_opt = torch.optim.AdamW(fresh.parameters())
    assert load_recovery(path, fresh, fresh_opt, cfg, 'uuid') == 17
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(model.weight, fresh.weight)
    assert fresh_opt.param_groups[0]['lr'] == .01
    assert fresh.runtime['entries_in_queue'] == 8
    with pytest.raises(ValueError, match='mismatch'):
        load_recovery(path, fresh, fresh_opt, cfg, 'other-run')


@pytest.mark.parametrize('override', ['model.patch_size=16', 'data.size_crops=[512,98]',
                                     'optim.clip_grad=3', 'train.max_steps=0', 'model.backbone=dinov3'])
def test_invalid_recipes_fail_before_loading_data(override):
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / 'configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-native-optim-100k-bs16-lr1e-6.yaml'),
                          OmegaConf.from_dotlist([override]))
    with pytest.raises(ValueError):
        validate(cfg)


def matched_config():
    return OmegaConf.load(ROOT / 'configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-100k-bs16-lr2e-4.yaml')


def test_dataset_selection_is_independent_of_trainer(monkeypatch):
    from src.data.loaders import DATASETS, make_dataset
    class AnotherDataset:
        def __init__(self, manifest, transform=None):
            self.manifest, self.transform = manifest, transform
    monkeypatch.setitem(DATASETS, 'ANOTHER', AnotherDataset)
    transform = object()
    result = make_dataset(OmegaConf.create({'name': 'ANOTHER', 'manifest': 'example.csv'}), transform)
    assert isinstance(result, AnotherDataset) and result.transform is transform
    with pytest.raises(ValueError, match='Unknown dataset'):
        make_dataset(OmegaConf.create({'name': 'UNKNOWN'}))


def test_dinov3_parameter_policy_and_schedules():
    from src.train.schedules import parameter_options
    spec = importlib.util.spec_from_file_location('dinov3_groups_reference', ROOT.parent / 'dinov3/dinov3/train/param_groups.py')
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    cfg = matched_config()
    # Check embedding, early/late blocks, norm, LayerScale and head roles.
    for name in ('patch_embed.proj.weight', 'patch_embed.proj.bias', 'cls_token', 'register_tokens',
                 'pos_embed', 'mask_token', 'blocks.0.attn.qkv.weight', 'blocks.11.mlp.fc1.weight',
                 'blocks.3.norm1.weight', 'blocks.7.ls1.gamma', 'norm.weight',
                 'projection_head.0.weight', 'projection_head.4.weight', 'projection_head.4.bias'):
        actual = parameter_options(name, 12, cfg, 'projection_head.4')
        alias = name.replace('register_tokens', 'storage_tokens')
        expected = reference.get_vit_lr_decay_rate(alias, .9, 12, force_is_backbone=True)
        if 'patch_embed' in name:
            expected *= .2
        assert actual['lr_multiplier'] == pytest.approx(expected)
        assert actual['wd_multiplier'] == float(not (name.endswith('bias') or any(k in name for k in ('norm', 'gamma', 'fourier_w'))))
        assert actual['is_last_layer'] == name.startswith('projection_head.4.')
    spec = importlib.util.spec_from_file_location('dinov3_schedule_reference', ROOT.parent / 'dinov3/dinov3/train/cosine_lr_scheduler.py')
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    lr = reference.CosineScheduler(2e-4, 1e-6, 100000, warmup_iters=750)
    wd = reference.CosineScheduler(.04, .4, 100000)
    ema = reference.CosineScheduler(.992, 1., 100000)
    class Optimizer:
        param_groups = [dict(is_head=False, lr_multiplier=.9, wd_multiplier=1.),
                        dict(is_head=True, lr_multiplier=1., wd_multiplier=0., is_last_layer=True)]
    optimizer = Optimizer()
    for step in (0, 1, 74, 75, 749, 750, 50000, 99999, 100000):
        values = set_schedule(optimizer, cfg, step)
        assert values['lr_backbone'] == pytest.approx(lr[step], abs=1e-15)
        assert values['lr_heads'] == pytest.approx(lr[step], abs=1e-15)
        assert values['weight_decay'] == pytest.approx(wd[step], abs=1e-15)
        assert values['momentum'] == pytest.approx(ema[step], abs=1e-15)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(lr[step] * .9)
        assert optimizer.param_groups[1]['lr'] == pytest.approx(0 if step < 75 else lr[step])


def test_matched_optimizer_clips_backbone_and_head_separately():
    from src.train.schedules import build_optimizer, clip_gradients
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.projection_head = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.GELU(), torch.nn.Linear(2, 1))
    class Model:
        model = Tiny()
    cfg = matched_config()
    model = Model()
    optimizer = build_optimizer(model, cfg)
    assert optimizer.defaults['betas'] == (.9, .999) and optimizer.defaults['eps'] == 1e-8
    for p in model.model.parameters():
        p.grad = torch.full_like(p, 10.)
    clip_gradients(model, cfg)
    for module in (model.model.blocks, model.model.projection_head):
        norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in module.parameters()]))
        assert float(norm) == pytest.approx(3., abs=1e-5)
    set_schedule(optimizer, cfg, 74)
    assert all(g['lr'] == 0 for g in optimizer.param_groups if g['is_last_layer'])
    set_schedule(optimizer, cfg, 75)
    assert all(g['lr'] > 0 for g in optimizer.param_groups if g['is_last_layer'])
