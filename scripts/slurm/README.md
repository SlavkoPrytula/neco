# NeCo microscopy training

For new or changed benchmark setups, follow the shared
[readiness checks](../../../dinov3/docs/setup/architecture.md#benchmark-readiness-checks)
and [logging validation](../../../dinov3/docs/setup/run-management.md#logging-validation).
Keep dated results below separate from those standing requirements.

Status, validated 2026-10-28: the role-based layout, dataset selection, native
recipe and new DINOv3 optimization recipe passed **14 CPU tests**, preflight
and H200 validation. The default launcher selects the DINOv3 optimization
recipe; the original native config is preserved unchanged.
Original upstream loss/model files remain unchanged. The selected recipe's
effective optimization schedule also matches saved DINOv2R run 3076277;
see the [comparison record](../../../dinov3/docs/experiments/benchmarks/pretrained-encoders.md#neco-dinov2r-benchmark-preparation--2026-10-28)
for the audit and current submission/cancellation status. The optimization policy was derived
from DINOv3 and also matches our DINOv2R baseline; the backbone is native DINOv2R.

Follow the shared [SSL naming convention](../../../dinov3/docs/protocols/configurations.md#ssl-config-naming).
The filename migration completed on 2026-10-28; commands below use the new paths.
Existing saved run folders and resolved configs retain their original names.

Latest [validation evidence](/gpfs/space/projects/PerkinElmer/dino_ssl/runs/diagnostics/neco/%5Bjob=3090974%5D-%5B2026-10-28_19-02-40%5D/validation.json):
132 diagnostic optimizer updates in reserved H200 allocation 3090974;
Slurm step `3090974.10` completed with exit `0:0` in 5m24s. This
repeats native parity/recovery/export and production checks after the moves,
then tests matched optimization at peak LR, exact model/optimizer reload and
final export, and 80 updates on the actual 100k schedule with eight workers
and recovery across the 75-step final-layer-freeze boundary. All checks pass;
peak matched-recipe GPU allocation was 34.74 GiB. This establishes execution,
not long-run convergence or optimal LR.

Earlier native-only GPU evidence: [validation.json](/gpfs/space/projects/PerkinElmer/dino_ssl/runs/diagnostics/neco/%5Bjob=3090974%5D-%5B2026-10-28_15-53-00%5D/validation.json),
[diagnostic runs](/gpfs/space/projects/PerkinElmer/dino_ssl/runs/diagnostics/neco/%5Bjob=3090974%5D-%5B2026-10-28_15-53-00%5D/),
and `outputs/slurm/job_3090974.log`, using the reserved H200 allocation.
Slurm step `3090974.9` completed with exit `0:0` in 2m29s.
The check covers native loss/181 student-gradient tensors/queue/EMA parity,
all 12 blocks updating, exact saved-state reload, 2+2-update recovery versus
two independent 4-update runs, final EMA export, W&B naming and 32 updates
with the real 100k schedule and eight workers. The queue reached 8192 entries;
losses/gradients remained finite. Peak allocated GPU memory: 34.74 GiB.
The short production diagnostic took 49.73 seconds including startup and
exports (10.30 images/s); this is not a long-run throughput guarantee.

Historical job 3090187 failed a numerically over-strict gradient assertion.
Native-vs-native repetition reproduced the failure without our wrapper:
GPU backward is nondeterministic, including CUDA bicubic position interpolation.
In the passing check, maximum absolute gradient differences were 3.20e-6 for
native repetition and 3.35e-6 for wrapper/native; maximum relative L2 errors
were 4.17e-6 and 3.88e-6. The test now checks both controls with fixed
`rtol=1e-4`, `atol=1e-5` and relative-L2 limit `1e-4`, retaining exact loss
repetition and queue equality. Recovery trajectories differed by at most
1.14e-6 in a parameter; exact checkpoint reload is checked separately from
nondeterministic subsequent computation. No native loss/model change was needed.

## Native method boundary

Upstream: `vpariza/NeCo`, commit `393216026b4d045e2e0b803f70a7d89d9a58937e`.
All existing files under `src/`, plus `experiments/utils.py`, are unchanged.
The preflight verifies their bytes against that commit and records their hashes.
`src/train/` is the orchestration layer: it inherits NeCo's original
loss, sorting, queue and ROI alignment; it calls the original
DINOv2R constructor, including student drop path 0.1 and teacher drop path 0.
The native projection head (three linear layers, hidden 2048, output 256), four
registers, independent initial heads, interpolation behaviour and native crop
transforms are retained. Optimizer grouping and EMA timing follow the selected
recipe: native NeCo or the explicit DINOv3 optimization adaptation. Frozen teacher parameters avoid an unused
autograd graph; the target remains the native EMA teacher.

Our changes are paired-brightfield loading, a continuous sample stream, an
update-addressed schedule, Slurm, W&B, portable EMA exports and full recovery.
The custom loop calls upstream `shared_step`; it does not use Lightning Trainer's
execution or the original Sacred/Neptune entry point. Lightning can support
update budgets; this custom loop was an implementation choice, not a framework
limitation. Switching runners would require separate recovery/EMA validation. Evaluation and
notebooks remain in `dinov3`; downstream NeCo evaluation is not validated by the
training test. Consumers must preserve NeCo's native interpolation/feature path,
not assume that an arbitrary official DINOv2 loader is forward-identical.

`src/models/dinov3_backbone.py` is an inactive, unvalidated adaptation draft.
It is never used by the DINOv2R recipe. DINOv3 integration is a separate validation
task, not part of the claim that the native NeCo architecture is preserved.

## Code organization

The integration now follows the coordinator's role-based structure:
`src/data/datasets/revvity.py`, `src/data/loaders.py`, `src/data/samplers.py`,
`src/train/{model,schedules,train}.py`, and `src/run/artifacts.py`.
Dataset selection and file checks stay out of the trainer. Register additional
image datasets in `src/data/loaders.py` and provide their own manifests/configs;
the native NeCo crop/correspondence transforms remain method-owned.
The former `src.benchmark.train` command is replaced by `src.train.train`.
Saved configs/checkpoints remain compatible; historical logs retain old paths.

## DINOv3 optimization recipe — default from 2026-10-28

User-requested recipe:
`configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-100k-bs16-lr2e-4.yaml`.
The original native recipe below is preserved unchanged for later experiments.
The new recipe has a distinct model name, W&B group and run directory.

| Setting | New DINOv3 optimization recipe | Preserved native NeCo recipe |
| --- | --- | --- |
| Encoder/head peak base LR | 2e-4 / 2e-4 (effective, no further batch scaling) | 1e-6 / 1e-4 |
| LR schedule | 750-update linear warmup, cosine to 1e-6 over 100k | No warmup, cosine to zero over 100k |
| Weight decay | 0.04 → 0.4 | 0.04 → 0.5 |
| EMA | 0.992 → 1, after optimizer update | 0.9995 → 1, before optimizer update |
| Layer LR decay / patch multiplier | 0.9 / 0.2 | None |
| Gradient clipping | Norm 3 independently for backbone and projection head | None |
| Final projection layer | LR zero for first 75 updates | No freezing |
| AdamW | Betas 0.9/0.999, epsilon 1e-8 | Same |

Reference: DINOv3's 100k recipe and
`dinov3/train/{train,param_groups,cosine_lr_scheduler}.py`, checked against the
selected run 2246959. Tests compare schedule values at warmup/freeze/end
boundaries and parameter multipliers against that source. DINOv2 registers map
to DINOv3's storage-token role for LR decay. DINOv3's named bias/norm/gamma
weight-decay exclusions replace native NeCo's dimension-based exclusions only
in this recipe. NeCo's final projection linear layer receives the analogous
75-update last-layer LR freeze; it is not a DINO prototype classifier.

There is no NeCo counterpart of DINO's teacher-temperature warmup or iBOT
masking; those are not added. Native FP32, 2×518/4×98 crop geometry, overlap,
augmentations, student drop path and loss remain unchanged. AdamW numerical
kernels/framework and the sampled data order can still differ. Thus this is
**matching optimization on a different objective**, not an otherwise-identical
DINOv3 reproduction or a fully controlled objective comparison. No LR search
or long-run convergence claim is implied by diagnostic validation.

## Preserved native recipe

`configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-native-optim-100k-bs16-lr1e-6.yaml` uses the frozen
well-v2 train manifest (8,938 fields), batch 16, seed 0, FP32, 100,000 updates.
Each image is the same shared-max-scaled RGB `[higher, higher, lower]` used by
DINOv3/DINOv2. No fluorescence or segmentation labels are used.

Native NeCo settings are retained: two 518px globals and four 98px locals;
area ranges 0.25–1 and 0.05–0.25; minimum intersection 0.01; encoder/head LR
1e-6/1e-4 decaying to zero; no warmup; AdamW betas 0.9/0.999; weight decay
0.04→0.5; EMA 0.9995→1; no gradient clipping or layerwise LR decay.
The horizon is extended to 100k updates on our dataset. This is not the published
COCO training duration or a loss-only comparison with DINO+iBOT.

### Comparison basis and hyperparameter audit — 2026-10-28

Source: [official ViT-B DINOv2R recipe](https://github.com/vpariza/NeCo/blob/393216026b4d045e2e0b803f70a7d89d9a58937e/experiments/configs/models/base/neco_dinov2r.yml),
[optimizer and loss implementation](https://github.com/vpariza/NeCo/blob/393216026b4d045e2e0b803f70a7d89d9a58937e/src/neco.py),
and [paper Appendix A.1 / Table 7(a)](https://arxiv.org/html/2408.11054v3#A1.SS1).
The code resolves the paper's generic “Adam” wording to **AdamW**. The native
Trainer specifies no mixed precision; this recipe explicitly uses full FP32.

| Setting | Effective benchmark value / justification |
| --- | --- |
| Encoder / head LR | `1e-6` / `1e-4`; same batch 16 as the pinned ViT-B/R recipe, no batch multiplier or layer decay |
| LR schedule | Native cosine to zero, no warmup, extended to 100k updates |
| Weight decay | Native cosine 0.04 → 0.5; zero for biases and one-dimensional parameters |
| AdamW | Native betas 0.9/0.999, epsilon 1e-8; no gradient clipping |
| Teacher | Native EMA 0.9995 → 1, updated after forward and before the student update |
| Spatial objective | Native 7×7 ROI alignment, 8192 reference queue, bitonic sorting, cosine similarity, steepness 100/100 |
| Views | Native 2×518 + 4×98; global area 0.25–1, local 0.05–0.25; overlap threshold 0.01 |
| Initialization | Same original DINOv2R weights as the DINO+iBOT baseline, four registers, full backbone trainable |

This is a **native-method recipe comparison at a common image budget**:
8,938 fixed training fields, batch 16, seed 0, 100k updates = 1.6M image
presentations. No microscopy hyperparameter search has been performed; values
are retained because they are published for this backbone and batch, and the
pipeline is checked for finite updates and recovery. This does not establish
optimal LR or long-run convergence on microscopy.

The 100k horizon replaces upstream's epoch-based COCO schedule; it represents
about 179 passes over our smaller pool. Table 7(a) also notes early stopping
of the reported DINOv2R model around epoch seven. We predeclare the final 100k
EMA for evaluation and do not choose a checkpoint from SSL loss. This is an
adaptation for our budget, not an exact published-checkpoint reproduction.

Against DINOv2R DINO+iBOT, data, starting encoder, image budget, FP32 and crop
counts/output sizes match. Loss, optimizer rates, EMA, crop-area ranges,
augmentations and sampled order are not all identical. Against DINOv3, the
backbone/pretraining, patch geometry and number of local views differ too.
Therefore this cannot isolate the objective or claim equal compute. Keep the
same frozen downstream decoder/split/seed/selection protocol per readout;
NeCo feature-loading integration in the coordinator still requires separate
validation before evaluation. Native recipe results do not replace the stricter
controlled-objective study in the research context.

The earlier unfinished draft copied DINO optimizer settings, used zero drop path
and different crop-area ranges; it was never submitted. Those choices were
removed on 2026-10-27 to keep native NeCo behaviour for this setup validation.

## Commands and outputs

Run all commands from `/gpfs/helios/home/prytula/Projects/dino-ssl/neco`.

```bash
bash scripts/slurm/train_revvity63x_neco.sh --check-only
/gpfs/helios/home/prytula/.conda/envs/neco/bin/python -m pytest -q tests/test_benchmark.py
# Reuse an authorized development allocation after checking it is running:
bash scripts/slurm/validate_revvity63x_neco.sh --job-id 3090974
# Alternatively, after confirmation of a new allocation:
sbatch scripts/slurm/validate_revvity63x_neco.sh
# Full training requires separate recipe/run confirmation:
sbatch scripts/slurm/train_revvity63x_neco.sh
# Original native optimizer recipe, retained for later experiments:
sbatch scripts/slurm/train_revvity63x_neco.sh --config-file configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-native-optim-100k-bs16-lr1e-6.yaml
sbatch scripts/slurm/train_revvity63x_neco.sh --run-dir /absolute/existing/run
```

Full launcher: account `revvity`, partition `gpu`, one H200-141GB, eight CPUs,
64GB host RAM, 96 hours. Validation: same resources, 20 minutes. GPU jobs never
run on login nodes. `NECO_PROJECT_ROOT`, `NECO_PYTHON`, `DINO_SSL_RUN_ROOT` are
optional launcher overrides. Configuration overrides use `key=value`; recovery
always reads the saved config and rejects overrides.

Research run directory:
`/gpfs/space/projects/PerkinElmer/dino_ssl/runs/ssl/revvity-63x/<config-stem>/[job=<id>]-[<UTC timestamp>]/`.
The default stem is `dinov2r-vitb14-neco-100k-bs16-lr2e-4`; the native recipe
uses `dinov2r-vitb14-neco-native-optim-100k-bs16-lr1e-6`.
It contains `config.yaml`, `run_metadata.json`, `metrics.json` (JSONL),
`logs/log.txt`, `wandb/`, full state in `checkpoints/resume/last.pth`, EMA exports
every 5k in `checkpoints/ema/`, and `model_final.pth` at 100k. Full state includes
student, teacher, optimizer, queue, RNG and completed update count. Resume retains
the run UUID and W&B name `<model-name>-s<seed>-j<first-job>`.
Online W&B logging uses project `dino-ssl` and existing user credentials.

Metadata records source revision/dirty state and native/integration hashes,
manifest/initial weight/export hashes, environment, per-attempt resources,
throughput, peak GPU memory and GPU-hours. Sample order resumes at consumed
fields. Worker augmentation RNG/prefetch is not replayed bitwise with workers>0;
the explicit deterministic recovery test uses zero workers.

Scheduler logs: `outputs/slurm/job_<id>.log`. Monitor with
`squeue -j <id>` and `tail -n 40 outputs/slurm/job_<id>.log`; cancel with
`scancel <id>`. Do not move an active scheduler log.

Validation writes only to shared `runs/diagnostics/neco/[job=...]-[...]`:
native-versus-wrapper forward/loss/student-gradient/queue/EMA parity, then two
updates plus recovery to four, versus four uninterrupted updates at batch 16
with real 518/98 crops. It checks all encoder blocks update, checkpoint identity,
offline W&B naming, saved metrics and final encoder export. An additional four-update uninterrupted repeat bounds CUDA variation. A separate
32-update diagnostic uses the unchanged 100k schedule, batch 16 and eight
workers, reaches a full reference queue and verifies finite gradients and
periodic exports. The matched-optimization check additionally exercises eight updates with a
compressed warmup/horizon to reach the peak LR, exact model/optimizer recovery
and final export, then 80 updates on the actual 100k schedule with recovery
across the 75-step final-layer-freeze boundary. Total: 132 diagnostic updates.
They are not a 100k run or evidence of long-run convergence. All diagnostic W&B runs stay offline.

Shared contracts: [workspace](../../../dinov3/docs/setup/benchmark-workspace.md),
[artifacts](../../../dinov3/docs/setup/run-management.md),
[dataset](../../../dinov3/docs/protocols/datasets.md).
