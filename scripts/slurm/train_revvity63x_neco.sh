#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=rev63x-neco-v2r
#SBATCH --output=./outputs/slurm/job_%j.log
#SBATCH --account=revvity
set -euo pipefail
PROJECT_ROOT=${NECO_PROJECT_ROOT:-/gpfs/helios/home/prytula/Projects/dino-ssl/neco}
PYTHON_BIN=${NECO_PYTHON:-/gpfs/helios/home/prytula/.conda/envs/neco/bin/python}
RUN_ROOT=${DINO_SSL_RUN_ROOT:-/gpfs/space/projects/PerkinElmer/dino_ssl/runs}
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
CHECK_ONLY=0
if [[ ${1:-} == --check-only ]]; then CHECK_ONLY=1; shift; fi
if [[ ${CHECK_ONLY} -eq 0 && -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Training requires sbatch; --check-only is safe on the login node' >&2
  exit 2
fi
if [[ ${1:-} == --run-dir ]]; then
  [[ $# -eq 2 ]] || { echo 'Recovery accepts only --run-dir <existing-run>' >&2; exit 2; }
  ARGS=(--run-dir "$(realpath -e "$2")")
else
  CONFIG=configs/ssl/revvity-63x/full/dinov2r-vitb14-neco-100k-bs16-lr2e-4.yaml
  if [[ ${1:-} == --config-file ]]; then
    [[ $# -ge 2 ]] || { echo 'Missing config path' >&2; exit 2; }
    CONFIG=$2; shift 2
  fi
  [[ -f ${CONFIG} ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
  ARGS=(--config-file "${CONFIG}")
  if [[ ${CHECK_ONLY} -eq 0 ]]; then
    RECIPE=$(basename "${CONFIG}" .yaml)
    STAMP=$(date -u +%Y-%m-%d_%H-%M-%S)
    ARGS+=(--output-dir "${RUN_ROOT}/ssl/revvity-63x/${RECIPE}/[job=${SLURM_JOB_ID}]-[${STAMP}]")
  fi
  ARGS+=("$@")
fi
if [[ ${CHECK_ONLY} -eq 1 ]]; then
  exec "${PYTHON_BIN}" -m src.train.train --check-only "${ARGS[@]}"
fi
srun "${PYTHON_BIN}" -m src.train.train "${ARGS[@]}"
