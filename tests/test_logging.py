"""W&B schema checks without starting a training run or contacting W&B."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from src.run.artifacts import RunRecord


def test_log_preserves_native_values_and_uses_training_step_axis(tmp_path):
    record = RunRecord.__new__(RunRecord)
    record.root = tmp_path
    record.cfg = SimpleNamespace(train=SimpleNamespace(batch_size=16))
    record.wandb = Mock()
    record.wandb.step = 0
    native = dict(completed_steps=100000, loss=3.8, lr_backbone=1e-6,
                  lr_heads=1e-6, weight_decay=.4, momentum=1., grad_norm=.2, attempt=2)
    record.log(native)
    assert json.loads((tmp_path / 'metrics.json').read_text()) == native
    record.wandb.log.assert_called_once_with({
        'train/completed_steps': 100000, 'train/total_loss': 3.8,
        'train/lr': 1e-6, 'train/head_lr': 1e-6, 'train/weight_decay': .4,
        'train/teacher_momentum': 1., 'train/grad_norm': .2, 'train/attempt': 2,
        'train/local_batch_size': 16, 'train/global_batch_size': 16}, step=100000)
    # Recovery can replay optimizer steps; the SDK event index must not rewind.
    record.wandb.step = 100001
    record.log({**native, 'completed_steps': 95010, 'attempt': 3})
    assert record.wandb.log.call_args.kwargs == {'step': 100001}
    assert record.wandb.log.call_args.args[0]['train/completed_steps'] == 95010
