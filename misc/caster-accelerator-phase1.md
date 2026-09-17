# CASTER-site heterogeneous executor: Phase 1 specification

## Objective

Replay CASTER-site tripartition updates and scores over data extracted by the
production parser while the CPU retains topology search and move selection.

Phase 1 validates the execution boundary. It does not use accelerator scores
to choose a topology.

## Portable architecture

The shared C++ contract contains:

- one ordered operation representation;
- production-sized `uint16_t` counters;
- partition, site, species, and individual mappings;
- a deterministic CPU executor and score reduction;
- validation rules that run before a batch reaches a device.

The HIP implementation is source-portable across AMD and NVIDIA targets. Each
target is compiled separately with its platform toolchain and architecture
flag. A CPU-only build remains available without HIP headers or runtime
dependencies.

The portable boundary must not expose vendor-specific stream, event, or
device types to CASTER-site parsing or topology code.

## Resident dataset

Static device buffers contain:

- encoded observations grouped by partition and species;
- partition offsets, site counts, and frequency vectors;
- species-to-individual ranges;
- site-to-partition indices.

Mutable counters remain on the device across ordered batches. A reusable tape
buffer is uploaded for each batch. Normal execution downloads only one
partial score per score operation and workgroup. Counter download is reserved
for validation and recovery.

The representation supports multiple individuals mapped to one species.
Missing observations are encoded explicitly and remain no-ops. Phase 1
rejects replicated site kernels (`nRep != 0`) until their weighting semantics
are represented in the shared contract.

## Determinism and validation

For the same parser output, initial colors, and ordered tape:

1. production `Tripartition` and the flattened CPU executor must have
   byte-identical counters;
2. CPU and HIP executors must return the same number and order of scores;
3. scores must pass explicit absolute and relative tolerances;
4. final CPU and HIP counters must be byte-identical;
5. invalid taxa, colors, operation kinds, and score counts must fail before a
   device launch;
6. parsing, chunk construction, missing data, and individual mappings must not
   change to improve accelerator scheduling.

Validation covers a list of alignments, multiple chunks, missing observations,
and at least one mapping with multiple individuals per species.

## Timings

Report separately:

- allocation;
- static upload;
- tape upload;
- kernel execution;
- score download;
- ordered host reduction;
- validation-only counter download.

Resident performance excludes allocation, static upload, and counter
validation download. Cold performance includes them.

## Phase boundary

Phase 1 is complete when replay passes on CPU and an AMD MI210 using real
CASTER-site partitions, and the same source is compile-checked for a supported
NVIDIA target when hardware is available.

Topology decisions, full-tree CPU/GPU RF comparison, quadrupartitions,
concurrent multi-GPU scheduling, and device-failure recovery remain later
phases.
