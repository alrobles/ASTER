#!/bin/bash -l
#SBATCH --job-name=caster-guides-det
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/a474r867/scratch/caster-guides-det-%j.out

set -euo pipefail

ROOT=${ASTER_ROOT:-${SLURM_SUBMIT_DIR}}
INPUT=${CASTER_GUIDE_INPUT:-/beegfs/a474r867/phylogenyAI/results/caster_backbone/benchmark/supermatrix-3000.fasta}
OUT_ROOT=${CASTER_GUIDE_OUT:-/home/a474r867/scratch/caster-guides-${SLURM_JOB_ID}}
GUIDE_WORKERS=${CASTER_GUIDE_WORKERS:-4}

if [ ! -s "$INPUT" ]; then
    echo "missing guide validation input: $INPUT" >&2
    exit 2
fi
if ! [[ "$GUIDE_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid CASTER_GUIDE_WORKERS: $GUIDE_WORKERS" >&2
    exit 2
fi
if [ "$GUIDE_WORKERS" -gt "${SLURM_CPUS_PER_TASK:-1}" ]; then
    echo "guide workers exceed allocated CPUs" >&2
    exit 2
fi

mkdir -p "$OUT_ROOT"
make -C "$ROOT" -f GNUmakefile caster-site-strict

run_case() {
    local label=$1
    local workers=$2
    /usr/bin/time -v -o "$OUT_ROOT/$label.time" \
        "$ROOT/bin/caster-site-strict" \
        -t 1 \
        --guide-workers "$workers" \
        -r 1 \
        -s 0 \
        --seed 233 \
        "$INPUT" \
        > "$OUT_ROOT/$label.tree" \
        2> "$OUT_ROOT/$label.log"
}

run_case sequential 1
run_case concurrent "$GUIDE_WORKERS"

sed '/#Guide workers:/d' "$OUT_ROOT/sequential.log" \
    > "$OUT_ROOT/sequential.normalized.log"
sed '/#Guide workers:/d' "$OUT_ROOT/concurrent.log" \
    > "$OUT_ROOT/concurrent.normalized.log"

cmp "$OUT_ROOT/sequential.tree" "$OUT_ROOT/concurrent.tree"
cmp "$OUT_ROOT/sequential.normalized.log" "$OUT_ROOT/concurrent.normalized.log"

echo "guide_validation=passed"
echo "guide_workers=$GUIDE_WORKERS"
sha256sum "$OUT_ROOT/sequential.tree" "$OUT_ROOT/concurrent.tree"
awk -F: '
    /Elapsed \\(wall clock\\) time/ {
        value=$2
        for (i=3; i<=NF; i++) value=value ":" $i
        gsub(/^[[:space:]]+/, "", value)
        print FILENAME "=" value
    }
    /Maximum resident set size/ {
        gsub(/^[[:space:]]+/, "", $2)
        print FILENAME "_max_rss_kb=" $2
    }
' "$OUT_ROOT/sequential.time" "$OUT_ROOT/concurrent.time"
