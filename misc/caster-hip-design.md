# CASTER-site heterogeneous executor: Phase 0 specification

## Objective

Measure whether an AMD HIP backend can accelerate CASTER-site without moving
topology search to the GPU or changing the scientific meaning of a chunk.

Phase 0 is an isolated tripartition microkernel. It is not wired into
`caster-site` and does not replace the current CPU implementation.

## Boundaries

The CPU retains:

- input parsing and chunk construction;
- topology representation and candidate generation;
- move selection and stopping criteria;
- ordered aggregation of partition scores;
- recovery after a device failure.

The device receives:

- an immutable taxon-major sequence matrix;
- an ordered operation tape;
- one frequency vector for the tested partition;
- initial or rebuilt tripartition counters.

The device returns:

- final counters for validation or recovery;
- one partial score per score operation and workgroup.

Sequence data and counters must remain resident across batches in the
integrated design. Phase 0 includes transfer time separately so that kernel
speed is not confused with end-to-end speed.

## Data model

The prototype uses one byte per observation:

```cpp
enum Base : uint8_t {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
    Missing = 4
};
```

The production CPU representation uses four one-hot bits. A byte is used in
Phase 0 to establish the execution contract before evaluating packed layouts.
Missing observations do not change counters.

Counters preserve the current implementation's storage type:

```cpp
uint16_t counts[site][3][4];
```

Every site is owned by one device thread. No atomics are needed because
threads never update another site's counters.

The operation tape is:

```cpp
enum class OperationKind : uint8_t {
    Update,
    ScoreTripartition
};

struct Operation {
    uint32_t taxon;
    int8_t from;
    int8_t to;
    OperationKind kind;
};
```

`from == -1` adds a previously unassigned taxon. `to == -1` removes a taxon.
Otherwise the operation moves the taxon between groups. The host must reject
group values outside `-1..2` and taxa outside the sequence matrix.

## Execution

One workgroup owns a contiguous tile of sites. Each thread:

1. loads its twelve counters;
2. interprets the complete tape in order;
3. reads one base for each update;
4. applies remove then add using `uint16_t`;
5. evaluates `scorePos` at every score operation;
6. participates in a deterministic workgroup reduction;
7. writes its final counters.

The kernel emits a matrix:

```text
partial_scores[score_operation][workgroup]
```

The host sums workgroups in ascending index order. Phase 0 does not use an
atomic global sum.

## Invariants

- Chunk boundaries and frequency vectors are inputs, never tuning knobs.
- Tape order is identical on CPU and GPU.
- Counter arithmetic uses the same `uint16_t` representation as CASTER-site.
- A missing base is a no-op.
- No output is accepted after a HIP launch, synchronization, or allocation
  failure.
- The host can rebuild device counters from colors and sequences.
- Fast and strict builds are measured separately.

## Validation

The benchmark generates a deterministic sequence matrix and a valid tape from
a seed. It runs the same tape through the CPU reference and HIP kernel.

Acceptance criteria:

1. final counters are byte-identical;
2. the number and order of scores are identical;
3. every score satisfies an explicit absolute-plus-relative tolerance;
4. repeated strict runs return the same counters and scores;
5. invalid tape entries fail before launch;
6. setup, host-to-device, kernel, device-to-host, and total times are reported;
7. the benchmark exits non-zero on any mismatch or HIP error.

`gpu_total_ms` is the sum of setup, allocation, upload, one average resident
kernel execution, download, and ordered host reduction. `benchmark_wall_ms`
also includes the warmup and all repeated timing iterations.

Full-search integration additionally requires RF zero between CPU and GPU
trees on the same taxa, seed, input, chunking, and operation order.

## Decision gates

Continue from Phase 0 to the CASTER executor only if:

- the MI210 kernel is correct;
- batched execution is faster than the portable CPU reference;
- projected end-to-end speedup is at least 2× after measured transfers;
- resident state fits with headroom on a 64 GB MI210.

Quadrupartition support follows tripartition validation. Multi-GPU scheduling
follows single-GPU integration. Neither is part of Phase 0.
