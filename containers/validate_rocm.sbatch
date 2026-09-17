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

apptainer exec --rocm --bind /beegfs:/beegfs "$SIF" \
    make -C "$ROOT" -f GNUmakefile caster-site-accelerator-hip HIP_ARCH=gfx90a

apptainer exec --rocm --bind /beegfs:/beegfs "$SIF" \
    "$ROOT/bin/caster-site-accelerator-hip" \
    -t 4 --seed 233 --accelerator-moves 240 \
    --accelerator-batch-moves 80 --accelerator-score-every 8 \
    "$ROOT/example/example.phylip"
