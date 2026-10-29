# NeCo benchmark implementation

Read `../dinov3/AGENTS.md` and its documentation map for the shared research,
HPC, dataset, artifact and development contracts. NeCo owns its training and
environment; `dinov3` owns evaluation, notebooks and aggregate research records.

Apply the shared [benchmark readiness checks](../dinov3/docs/setup/architecture.md#benchmark-readiness-checks)
and [logging validation](../dinov3/docs/setup/run-management.md#logging-validation)
before declaring an integration ready and after its first authorized launch.

Read `scripts/slurm/README.md` before launching. Preserve the native NeCo loss
and crop correspondence. Register tokens must remain outside spatial grids.
The active backbone is native DINOv2R. Keep data in `src/data/`, training in
`src/train/` and artifacts in `src/run/`. The default recipe matches DINOv3
optimization; preserve the separate native-NeCo recipe and label results accordingly. The DINOv3 wrapper is an inactive,
unvalidated draft; a DINOv3 port is a separate adaptation, not a native benchmark.
Record exact upstream and imported backbone revisions. GPU execution and
checkpoint recovery require explicit Slurm validation, not only CPU tests.
