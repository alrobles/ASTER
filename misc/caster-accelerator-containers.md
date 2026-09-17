# CASTER-site accelerator containers

The accelerator benchmark uses two pinned Apptainer images:

| Target | Base | Compiler | Runtime binding |
| --- | --- | --- | --- |
| AMD MI210 | `rocm/dev-ubuntu-24.04:6.4.3-complete` | `hipcc` | `--rocm` |
| NVIDIA A100 | `nvidia/cuda:12.4.1-devel-ubuntu22.04` | `nvcc` with ROCm 6.4.3 HIP headers | `--nv` |

The NVIDIA image does not depend on a `hipcc` NVIDIA redistribution. ROCm's
HIP headers map the runtime API to CUDA when the source is compiled by `nvcc`
with `__HIP_PLATFORM_NVIDIA__`. AMD and NVIDIA still produce separate binaries.

## Build

The images can be built without a GPU:

```bash
containers/build_image.sh rocm /path/to/caster-rocm.sif
containers/build_image.sh cuda /path/to/caster-cuda.sif
```

The wrapper tries unprivileged fakeroot first, uses a writable Apptainer cache,
and falls back to the site's plain build mode.

## Compile

```bash
apptainer exec caster-rocm.sif \
  make -f GNUmakefile caster-site-accelerator-hip HIP_ARCH=gfx90a

apptainer exec caster-cuda.sif \
  make -f GNUmakefile caster-site-accelerator-cuda CUDA_ARCH=sm_80
```

## Run

```bash
apptainer exec --rocm --bind /beegfs:/beegfs caster-rocm.sif \
  ./bin/caster-site-accelerator-hip [options] alignment

apptainer exec --nv --bind /beegfs:/beegfs caster-cuda.sif \
  ./bin/caster-site-accelerator-cuda [options] alignment
```

The CPU reference remains independent of either GPU runtime:

```bash
make -f GNUmakefile caster-site-accelerator-reference
```

The Slurm validation jobs compile and execute the same ordered operation tape
on MI210 and A100:

```bash
sbatch containers/validate_rocm.sbatch
sbatch containers/validate_cuda.sbatch
```

Both jobs require `validation=passed`. This validates resident counter and
score replay. It does not establish equivalence of complete inferred trees.
