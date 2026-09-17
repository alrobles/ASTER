#!/bin/bash -l
#SBATCH --job-name=caster_ctr_nv
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/a474r867/scratch/caster-ctr-nv-%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
SIF="${CASTER_SIF:-/beegfs/a474r867/ASTER/images/caster-cuda.sif}"

apptainer exec --bind /beegfs:/beegfs "$SIF" \
    make -C "$ROOT" -f GNUmakefile caster-site-accelerator-reference

apptainer exec --bind /beegfs:/beegfs "$SIF" \
    "$ROOT/bin/caster-site-accelerator-reference" \
    -t 4 --seed 233 --accelerator-moves 240 \
    --accelerator-batch-moves 80 --accelerator-score-every 8 \
    "$ROOT/example/example.phylip"

apptainer exec --nv --bind /beegfs:/beegfs "$SIF" \
    make -C "$ROOT" -f GNUmakefile caster-site-accelerator-cuda \
    CUDA_ARCH=sm_80

apptainer exec --nv --bind /beegfs:/beegfs "$SIF" \
    "$ROOT/bin/caster-site-accelerator-cuda" \
    -t 4 --seed 233 --accelerator-moves 240 \
    --accelerator-batch-moves 80 --accelerator-score-every 8 \
    "$ROOT/example/example.phylip"
