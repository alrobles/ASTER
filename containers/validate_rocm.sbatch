#!/bin/bash -l
#SBATCH --job-name=caster_ctr_amd
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/a474r867/scratch/caster-ctr-amd-%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
SIF="${CASTER_SIF:-/beegfs/a474r867/ASTER/images/caster-rocm.sif}"
CPU_DIR="${TMPDIR:-/tmp}/caster-validation-${SLURM_JOB_ID}"
CPU_BIN="${CPU_DIR}/caster-site-accelerator-reference"
mkdir -p "$CPU_DIR"
trap 'rm -f "$CPU_BIN"; rmdir "$CPU_DIR" 2>/dev/null || true' EXIT

apptainer exec --bind /beegfs:/beegfs --bind "$CPU_DIR:$CPU_DIR" "$SIF" \
    make -C "$ROOT" -f GNUmakefile caster-site-accelerator-reference \
    ACCELERATOR_REFERENCE_BIN="$CPU_BIN"

apptainer exec --bind /beegfs:/beegfs --bind "$CPU_DIR:$CPU_DIR" "$SIF" \
    "$CPU_BIN" \
    -t 4 --seed 233 --accelerator-moves 240 \
    --accelerator-batch-moves 80 --accelerator-score-every 8 \
    --accelerator-production-tape \
    "$ROOT/example/example.phylip"

apptainer exec --rocm --bind /beegfs:/beegfs "$SIF" \
    make -C "$ROOT" -f GNUmakefile caster-site-accelerator-hip HIP_ARCH=gfx90a

apptainer exec --rocm --bind /beegfs:/beegfs "$SIF" \
    "$ROOT/bin/caster-site-accelerator-hip" \
    -t 4 --seed 233 --accelerator-moves 240 \
    --accelerator-batch-moves 80 --accelerator-score-every 8 \
    --accelerator-production-tape \
    "$ROOT/example/example.phylip"
