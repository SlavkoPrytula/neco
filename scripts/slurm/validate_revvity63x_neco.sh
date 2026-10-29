#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --hint=nomultithread
#SBATCH --job-name=validate-neco-v2r
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
STEP_ARGS=()
if [[ ${1:-} == --job-id ]]; then
  [[ $# -eq 2 && $2 =~ ^[0-9]+$ ]] || { echo 'Usage: validate_revvity63x_neco.sh --job-id <reserved-job>' >&2; exit 2; }
  VALIDATION_JOB=$2
  STEP_ARGS=(--jobid="${VALIDATION_JOB}" --exact --nodes=1 --ntasks=1
             --cpus-per-task=8 --gres=gpu:1 --mem=64G --time=00:20:00
             --output="${PROJECT_ROOT}/outputs/slurm/job_${VALIDATION_JOB}.log" --open-mode=append)
elif [[ $# -eq 0 ]]; then
  : "${SLURM_JOB_ID:?Use sbatch or --job-id with an existing allocation}"
  VALIDATION_JOB=${SLURM_JOB_ID}
else
  echo 'Usage: validate_revvity63x_neco.sh [--job-id <reserved-job>]' >&2; exit 2
fi
STAMP=$(date -u +%Y-%m-%d_%H-%M-%S)
DESTINATION=${RUN_ROOT}/diagnostics/neco/[job=${VALIDATION_JOB}]-[${STAMP}]
echo "Validation artifacts: ${DESTINATION}"
srun "${STEP_ARGS[@]}" "${PYTHON_BIN}" tests/gpu_validation.py --output-dir "${DESTINATION}"
