#!/bin/bash -l
#SBATCH --job-name=caster_hip_p0
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/home/a474r867/scratch/caster-hip-p0-%j.out

set -euo pipefail
exec 2>&1

REPO_ROOT=${SLURM_SUBMIT_DIR:?Submit this job from the ASTER repository root}
ROCM=${ROCM:-/kuhpc/sw/rocm/6.4.3}
REPEATS=${REPEATS:-3}
TRACE="TRACE_CASTER_HIP_PHASE0_${SLURM_JOB_ID}"

source /kuhpc/sw/lmod/lmod/init/profile
ml purge
ml load rocm/6.4.3
cd "$REPO_ROOT"

echo "trace=$TRACE"
echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
echo "rocm=$("$ROCM/bin/hipcc" --version | head -1)"
"$ROCM/bin/rocm-smi" --showproductname 2>/dev/null | head -4 || true

make caster-site-hip-bench HIPCC="$ROCM/bin/hipcc" HIP_ARCH=gfx90a

run_regime() {
    local name=$1
    shift
    local repeat
    for repeat in $(seq 1 "$REPEATS"); do
        echo "=== regime=$name repeat=$repeat ==="
        bin/caster-site-hip-bench "$@"
    done
}

run_regime smoke \
    --sites 4096 --taxa 32 --moves 24 --score-every 3 \
    --block-size 256 --iterations 3 --seed 233

run_regime mini_147 \
    --sites 47745 --taxa 147 --moves 30 --score-every 3 \
    --block-size 256 --iterations 7 --seed 233

run_regime mammal_4353 \
    --sites 280189 --taxa 4353 --moves 30 --score-every 3 \
    --block-size 256 --iterations 7 --seed 233

echo "finished=$(date --iso-8601=seconds)"
echo "status=passed"
