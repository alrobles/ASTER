# CASTER-site heterogeneous search integration: Phase 2 specification

## Objective

Use the resident executor from Phase 1 for production tripartition updates and
scores while preserving CASTER-site's topology search, random stream, move
order, and CPU fallback.

Phase 2 accelerates the existing heuristic. It does not batch NNI decisions,
change guide-tree sampling, or use an external reference tree.

## Scope

Phase 2 is split into three independently verifiable increments:

1. route one production `PlacementAlgorithm` through a private resident CPU
   search state;
2. route the same ordered operation stream through one HIP device;
3. execute independent guide searches concurrently and merge their
   tripartitions in original guide-index order.

Branch-by-site placement kernels, concurrent insertion of multiple taxa,
batched NNI, beam search, and multi-node scheduling remain later phases.

## Components

### Immutable dataset

One shared dataset contains:

- encoded observations;
- statistical partition frequencies, weights, and offsets;
- species-to-individual ranges;
- site-to-partition mappings;
- taxon hashes and names used by topology code.

Parsing, informative-site filtering, individual mappings, missing-data
semantics, and partition weights remain unchanged.

Statistical partitions are not execution units. CPU workers and GPU blocks may
shard sites from one statistical partition while every site continues to use
that partition's original frequencies and weight. Accelerator scheduling must
not change `--chunk`, recompute frequencies, or split an input alignment into
new biological partitions.

### Private search state

Every active guide or placement search owns:

- one color per taxon;
- one tripartition counter array;
- cached score state;
- topology nodes, parents, children, and placement dynamic-programming state;
- its precomputed insertion order and random decisions.

No mutable counter or color may remain in the shared initializer on the
accelerated path.

### Deterministic search plan

Before workers start, the legacy pseudorandom generator produces a
single-threaded `SearchPlan` containing every decision that would otherwise
consume global random state:

- initial taxon hashes;
- guide taxon subsets and insertion orders;
- subsampling subsets and insertion orders;
- stable search and merge indices.

Workers consume this immutable plan and never call the global random generator.
Plan serialization includes a version, seed, input hash, and plan hash so CPU,
AMD, and NVIDIA runs can prove that they executed the same search.

### Ordered operation queue

The topology code emits:

```text
Update(taxon, from, to)
ScoreTripartition
```

in the same order as the existing `ThreadPool` task queue. The first result
request flushes the pending queue. Update results occupy their original queue
positions and return zero; score results preserve their original positions.

This queue is the correctness boundary shared by the resident CPU and HIP
executors.

### Executor factory

An executor factory creates one private mutable state per
`PlacementAlgorithm`. The factory may share the immutable dataset but must not
share colors, counters, pending operations, result queues, or failure state.

Supported implementations are:

- production CPU;
- flattened resident CPU reference;
- HIP compiled for AMD;
- HIP compiled for NVIDIA.

The topology layer must not include vendor-specific device, stream, or event
types.

## Determinism

For a fixed input, seed, commit, arguments, and hardware profile:

1. generate and hash the complete search plan before concurrent execution;
2. assign every guide a stable index;
3. keep every guide's mutable state private;
4. store results by guide index, independent of completion order;
5. merge tripartitions in ascending guide-index order;
6. preserve the current sequential order for taxon insertion and NNI moves;
7. reduce site and device shards in a documented fixed order.

Scores close enough to change a branch or move decision must be recomputed by
the production CPU backend before applying that decision.

## Failure and fallback

Host colors advance only after a batch completes successfully. If a device
batch fails before its results are consumed:

1. mark that device state unusable;
2. rebuild private CPU counters from the last committed host colors;
3. replay the same pending operation queue on CPU;
4. continue without changing operation or result order;
5. record the failure, fallback point, and backend transition.

A failure after a topology decision has consumed partial results is fatal; the
run must not silently continue from an ambiguous state.

## Resource limits

Before starting a concurrent batch, the scheduler computes:

- immutable dataset bytes per device;
- mutable counter and queue bytes per search;
- host topology bytes per search;
- maximum resident searches under an explicit memory reserve;
- CPU worker and device concurrency caps.

Oversubscription must reduce concurrency or select CPU fallback. It must not
permit allocator failure as normal flow.

## Instrumentation

Report separately:

- parsing and informative-site filtering;
- immutable dataset construction and upload;
- guide search;
- branch assignment;
- branch-group placement;
- sequential abnormal-taxon placement;
- NNI;
- subsampling;
- tape upload, kernel, score download, and host reduction;
- fallback and recovery;
- peak host and device memory.

Every report includes raw columns, informative sites, taxa, partitions,
threads, device model, backend, seed, commit, and final tree hash.

## Acceptance criteria

### Increment 1: private CPU state

- production and flattened CPU operation traces are identical;
- one statistical partition can be divided across multiple execution shards
  without changing its score;
- counters are byte-identical after every checked flush;
- score vectors pass explicit absolute and relative tolerances;
- final topology has RF 0 against production CPU;
- two concurrent guide searches pass thread and address sanitizers;
- repeated runs produce identical tree and trace hashes.

### Increment 2: one GPU

- CPU and HIP consume identical ordered traces;
- final counters are byte-identical on representative real subsets;
- ambiguous numerical decisions use the production CPU fallback;
- final topology has RF 0 against production CPU for every validation case;
- end-to-end timing includes all transfers, reductions, and CPU coordination;
- both MI210 and A100 pass using their separate Apptainer images.

### Increment 3: concurrent guides

- precomputed guide orders match the sequential baseline;
- guide tripartition hashes match by guide index;
- deterministic ordered merging yields the same optimal tree and score;
- final topology has RF 0 against the sequential accelerated path;
- speedup is reported against elapsed time, CPU-hours, GPU-hours, and energy
  when available.

## Validation ladder

Use both simulated data with a known species tree and real codon-aware
supermatrix subsets:

```text
taxa:     250 -> 1,000 -> 4,353
columns:  300 -> 3,000 -> 30,000
```

Record informative sites after filtering. Simulations vary incomplete lineage
sorting, missing data, locus heterogeneity, and alignment difficulty.

Compare topology, objective score, quartet score, replicate stability,
support, elapsed time, peak memory, and energy. Robinson-Foulds comparisons
are always performed on shared taxa and report the shared-taxon count.

## Out of scope

Upham or any other published tree is a post-inference comparison only. It is
never a guide or constraint.

Batched best-improvement NNI and beam or population search are new heuristics.
They require separate names, outputs, and validation against the exact
accelerated path.
