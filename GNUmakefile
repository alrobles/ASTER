include makefile

HIPCC ?= hipcc
HIP_ARCH ?= gfx90a

.PHONY: caster-site-portable caster-site-strict caster-site-kernel-reference caster-site-hip-bench caster-site-accelerator-reference caster-site-accelerator-hip

caster-site-portable: dir
	g++ -std=gnu++17 -O3 -ffast-math -pthread src/caster-site.cpp -o bin/caster-site-portable

caster-site-strict: dir
	g++ -std=gnu++17 -O2 -fno-fast-math -ffp-contract=off -pthread src/caster-site.cpp -o bin/caster-site-strict

caster-site-kernel-reference: dir
	g++ -std=gnu++17 -O3 -fno-fast-math -ffp-contract=off src/caster-site-hip-bench.cpp -o bin/caster-site-kernel-reference

caster-site-hip-bench: dir
	$(HIPCC) -std=c++17 -O3 -fno-fast-math -ffp-contract=off --offload-arch=$(HIP_ARCH) src/caster-site-hip-bench.cpp -o bin/caster-site-hip-bench

caster-site-accelerator-reference: dir
	g++ -std=gnu++17 -O2 -fno-fast-math -ffp-contract=off -pthread src/caster-site-accelerator-bench.cpp -o bin/caster-site-accelerator-reference

caster-site-accelerator-hip: dir
	$(HIPCC) -std=gnu++17 -O2 -fno-fast-math -ffp-contract=off -pthread --offload-arch=$(HIP_ARCH) src/caster-site-accelerator-bench.cpp -o bin/caster-site-accelerator-hip
