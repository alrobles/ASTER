include makefile

HIPCC ?= hipcc
HIP_ARCH ?= gfx90a
NVCC ?= nvcc
CUDA_ARCH ?= sm_80
HIP_NVIDIA_INCLUDE ?= /opt/hip/include
ACCELERATOR_REFERENCE_BIN ?= bin/caster-site-accelerator-reference

.PHONY: caster-site-portable caster-site-strict caster-site-guide-validation caster-site-kernel-reference caster-site-hip-bench caster-site-accelerator-reference caster-site-accelerator-sanitize caster-site-accelerator-tsan caster-site-accelerator-hip caster-site-accelerator-cuda

caster-site-portable: dir
	g++ -std=gnu++17 -O3 -ffast-math -pthread src/caster-site.cpp -o bin/caster-site-portable

caster-site-strict: dir
	g++ -std=gnu++17 -O2 -fno-fast-math -ffp-contract=off -pthread src/caster-site.cpp -o bin/caster-site-strict

caster-site-guide-validation: caster-site-strict
	tmpdir=$$(mktemp -d); trap 'rm -rf "$$tmpdir"' EXIT; \
	./bin/caster-site-strict -t 1 --guide-workers 1 -r 1 -s 0 --seed 233 example/example.phylip > "$$tmpdir/sequential.tree" 2> "$$tmpdir/sequential.log"; \
	./bin/caster-site-strict -t 1 --guide-workers 2 -r 1 -s 0 --seed 233 example/example.phylip > "$$tmpdir/concurrent.tree" 2> "$$tmpdir/concurrent.log"; \
	sed '/#Guide workers:/d' "$$tmpdir/sequential.log" > "$$tmpdir/sequential.normalized.log"; \
	sed '/#Guide workers:/d' "$$tmpdir/concurrent.log" > "$$tmpdir/concurrent.normalized.log"; \
	cmp "$$tmpdir/sequential.tree" "$$tmpdir/concurrent.tree"; \
	cmp "$$tmpdir/sequential.normalized.log" "$$tmpdir/concurrent.normalized.log"

caster-site-kernel-reference: dir
	g++ -std=gnu++17 -O3 -fno-fast-math -ffp-contract=off src/caster-site-hip-bench.cpp -o bin/caster-site-kernel-reference

caster-site-hip-bench: dir
	$(HIPCC) -std=c++17 -O3 -fno-fast-math -ffp-contract=off --offload-arch=$(HIP_ARCH) src/caster-site-hip-bench.cpp -o bin/caster-site-hip-bench

caster-site-accelerator-reference: dir
	g++ -std=gnu++17 -O2 -fno-fast-math -ffp-contract=off -pthread src/caster-site-accelerator-bench.cpp -o $(ACCELERATOR_REFERENCE_BIN)

caster-site-accelerator-sanitize: dir
	g++ -std=gnu++17 -O1 -g -fno-omit-frame-pointer -fno-fast-math -ffp-contract=off -pthread -fsanitize=address,undefined src/caster-site-accelerator-bench.cpp -o bin/caster-site-accelerator-sanitize

caster-site-accelerator-tsan: dir
	g++ -std=gnu++17 -O1 -g -fno-omit-frame-pointer -fno-fast-math -ffp-contract=off -pthread -fsanitize=thread src/caster-site-accelerator-bench.cpp -o bin/caster-site-accelerator-tsan

caster-site-accelerator-hip: dir
	$(HIPCC) -std=gnu++17 -O2 -fno-fast-math -ffp-contract=off -pthread --offload-arch=$(HIP_ARCH) src/caster-site-accelerator-bench.cpp -o bin/caster-site-accelerator-hip

caster-site-accelerator-cuda: dir
	$(NVCC) -std=c++17 -O2 --fmad=false -D__HIP_PLATFORM_NVIDIA__ -D__GLIBCXX_TYPE_INT_N_0=__int128 -D__GLIBCXX_BITSIZE_INT_N_0=128 -I$(HIP_NVIDIA_INCLUDE) -arch=$(CUDA_ARCH) -x cu -Xcompiler=-pthread,-fno-fast-math,-ffp-contract=off src/caster-site-accelerator-bench.cpp -o bin/caster-site-accelerator-cuda
