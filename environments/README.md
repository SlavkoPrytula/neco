# NeCo training environment

Installed prefix: `/gpfs/helios/home/prytula/.conda/envs/neco`.
Python 3.11.18, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124,
xFormers 0.0.29.post1, Lightning 2.5.0.post0. This environment is independent of
`dinov2` (torch 2.0.1+cu118) and `dino-ssl` (torch 2.11.0+cu128).

The direct requirements are in `neco.txt`; the installed package lock is
`neco.lock.txt`. Recreate with:

```bash
conda create -n neco python=3.11.18 pip -y
conda run -n neco python -m pip install -r environments/neco.txt
conda run -n neco python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r environments/neco.lock.txt
conda run -n neco python -m pip check
```

Dependency/import checks pass. Hardware validation status is tracked in the
[training guide](../scripts/slurm/README.md). The benchmark launcher uses W&B;
the original Sacred/Neptune CLI and its external datasets are not configured.
Its source remains unchanged. No credentials are stored in this repository.
