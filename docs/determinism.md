# Determinism contract

EMEL's text generation runtime guarantees **bitwise determinism**: the same
model, the same input, and the same configuration produce bit-for-bit
identical logits and token streams on every run. This is a stated, tested,
and gated contract, not an accident of the current implementation.

This is a deliberate differentiator from llama.cpp/ggml, whose intra-op
thread chunking splits one float reduction across `n_threads` partial sums,
so results change with the thread count. EMEL's parallel routes never split a
float reduction: results are independent of thread scheduling and invariant
to the parallel lane count.

## 1. What is guaranteed

For a fixed model file, prompt/messages, generation configuration
(`selection_mode`, token budget, formatter mode, stop sequences), and sampler
callback chain, on the **same host and the same build**:

- **Bitwise-identical logits.** Every decode step materializes the identical
  logits vector, byte for byte, on every run.
- **Bitwise-identical token streams.** The selected token ids and the
  detokenized output bytes are identical on every run.
- **Repeat-, session-, and process-invariant.** The guarantee holds across
  repeated `generate` dispatches on one session, across freshly constructed
  sessions (new generator, new KV blocks, new buffers), and across separate
  process invocations (fresh address space; nothing in the numeric path
  depends on pointer values, ASLR, wall-clock time, or environment).
- **Thread-schedule- and lane-count-invariant.** Parallel execution never
  changes results: the fork/join parallel matmul route is bitwise identical
  to serial dispatch and bitwise invariant to the number of row-slice lanes
  (see section 4).
- **Route-consistent argmax.** The fused `preselected_argmax` decode route
  and the materialized `sample_logits` route (with an argmax sampler
  callback) select the same tokens and produce the same output bytes; the
  determinism gate asserts this equivalence on a maintained fixture.

### What is NOT guaranteed

- **Cross-host / cross-ISA identity.** Kernel variants differ by
  architecture (`kernel/aarch64` vs `kernel/x86_64`) and by host feature
  tuning (`-mcpu=native+dotprod+i8mm`, AVX2/FMA/F16C probes in
  `CMakeLists.txt`). Different hosts may produce different (each internally
  deterministic) bit patterns.
- **Cross-build identity.** A different toolchain, optimization level, or
  kernel-variant selection may change float contraction and therefore bits.
  The contract is per host + build.
- **Identity across configuration changes.** Changing the attention mode,
  operand preparation (packed/prepared quant formats), context capacity, or
  formatter contract legitimately changes the computed values.

## 2. Seed and RNG policy

**The EMEL runtime contains no random number generator.** There is no RNG
facility anywhere under `src/` (no `<random>`, no `rand`, no xorshift/PCG
state); nothing in the runtime reads wall-clock time or entropy during
dispatch (time injection is an explicit SML rule).

Token selection has exactly two maintained modes
(`emel::text::generator::selection_mode` in
`src/emel/text/generator/events.hpp`):

- `preselected_argmax` — the runtime's fused argmax route. Argmax scans are
  strict `>` comparisons in ascending row order (`op_mul_mat_argmax` /
  `op_argmax` kernels in `src/emel/kernel/detail.hpp`), so ties resolve to
  the **lowest token id**, deterministically.
- `sample_logits` — the materialized route. The logits sampler machine
  (`src/emel/logits/sampler`) copies the logits into a candidate array in
  ascending token-id order and then invokes the **externally injected**
  sampler callback chain (`emel::logits::sampler::fn`, bound at
  `event::initialize.sampler_fns`) in fixed order. All stochastic sampling —
  and therefore all seeding — lives in the caller's callbacks. If the caller
  injects a seeded PRNG and seeds it identically, generation is bitwise
  reproducible; the runtime adds no entropy of its own.

The `seed` field in benchmark workload manifests
(`tools/bench/generation_variants/**.json`) is recorded metadata for
reference-lane comparability. The EMEL lane never consumes it: the maintained
workloads use `sampling_id=argmax_v1`, which is seed-free.

## 3. Float reduction-ordering policy

Every floating-point reduction in the generation path executes in a **fixed,
statically determined order**:

- Which kernel variant runs (dtype, architecture, packed/prepared operand
  class, attention mode) is selected by route guards in the state-machine
  transition tables at dispatch time — a pure function of model contract and
  host kernel kind, not of thread timing.
- Within a kernel, each output element's reduction (dot product over the
  K dimension, softmax accumulation, norm sums) runs serially on a single
  thread in the kernel's fixed loop order.
- There is no atomic float accumulation, no work-stealing partial-sum
  merging, and no runtime-thread-count-dependent chunking anywhere in `src/`.
- Argmax reductions over the vocabulary are never sliced across lanes: the
  fused argmax always dispatches as a single serial kernel event
  (`matmul_vector_argmax` in `src/emel/text/generator/detail.hpp` runs on the
  primary kernel actor only).

## 4. Parallel-lane policy and lane-count invariance

The maintained runtime has five fork/join or overlap surfaces. Every compute
fork joins within the owning RTC dispatch, and none reorders a float reduction:

1. **Kernel-owned row-sliced matmul**
   (`src/emel/kernel/matmul/**`). A logical `mul_mat` is routed by explicit
   state-machine guards to serial, 2-, 4-, or 8-lane effects. The owner runs
   lane zero while a construction-time-sized `fork_join_lane_pool` runs the
   remaining lanes. Slices are contiguous, aligned to the packed row group,
   write disjoint output rows, and share read-only operands. Each output
   element's reduction stays wholly within one lane in the kernel's fixed
   order, so results are bitwise invariant across the supported lane counts.
   The pool starts exactly `active_lanes - 1` workers on budgeted runtime paths;
   unused capacity does not create runnable or spinning threads.

   Invariance is proven directly:
   `tests/text/generator/determinism_tests.cpp` sweeps slice counts
   {1, 2, 3, 5, 8} over f32/q8_0/q4_k/q6_k GEMV and f32 GEMM and asserts
   bitwise equality with the unsliced dispatch;
   `tests/kernel/matmul_tests.cpp` asserts serial vs
   parallel fork/join bit-exactness on the production dispatch helper.

2. **Moshi temporal-attention head lanes**
   (`src/emel/speech/predictor/moshi/executor/actions.hpp`). The same budgeted
   lane pool dispatches disjoint head ranges to independent attention actors.
   Each head writes a disjoint output interval and performs its softmax and
   weighted-value reductions locally in fixed order. The 1/2/4/8-lane result
   and cache state are compared exactly by
   `tests/speech/predictor/moshi_lifecycle_tests.cpp`.

3. **Speech-generator frame wavefront**
   (`src/emel/speech/generator/actions.hpp`). A fixed two-worker stage pool can
   overlap encoder and decoder actors with the owner-thread Moshi middle stage.
   The actors own disjoint frame lanes, preserve typed attribution, and join
   before the action returns. Serial stage mode creates no stage pool. Lifecycle,
   overlap, drain, attribution, and deterministic-trace contracts are covered by
   `tests/speech/generator/wavefront_lifecycle_tests.cpp`.

4. **Decode wavefront lanes**
   (`src/emel/text/generator/decode_wavefront/actions.hpp`). Forks
   *independent generation sessions* (each with its own graph, KV state, and
   buffers) across the pool. There is no shared mutable float state between
   lanes, so each session's stream is unchanged by running next to others
   (`tools/decode_wavefront_eval` asserts sequential == parallel outputs).

5. **Tensor-window stream I/O**
   (`src/emel/model/tensor/window/actions.hpp`). Overlaps weight prefetch
   I/O with compute; it moves bytes and performs no float arithmetic.

**Current invariance status: no known gaps.** No parallel route performs a
cross-lane float reduction; all softmax, norm, dot-product, and weighted-value
reductions remain local to one output element or attention head. If a future
parallel route must split a reduction (e.g., intra-op flash-attention
K-splitting), it MUST either merge partial results in a fixed, lane-count-
independent order or record the resulting per-lane-count-only guarantee here,
with the offending code path, before it lands.

## 5. How the contract is gated

- `tests/text/generator/determinism_tests.cpp` — doctest proofs of
  slice-count invariance and parallel fork/join repeatability (runs in the
  `generator_and_runtime` shard of `emel_tests`).
- `tools/determinism_check` — an EMEL-only harness (no llama.cpp/ggml
  linkage) that loads the maintained `tests/models/LFM2.5-230M-Q8_0.gguf`
  fixture through the production machines and runs the same `generate`
  request repeatedly in both selection modes, checksumming the output bytes,
  the selected token ids, and every materialized logits vector (FNV-1a via
  the public sampler seam). It hard-fails unless every repeat and a freshly
  constructed session are bitwise identical, and unless the fused-argmax and
  materialized routes agree on the token stream and output bytes.
- `scripts/check_determinism.sh` — builds the harness and runs it in two
  separate processes; hard-fails unless both processes emit identical
  `determinism_evidence` lines (cross-process determinism).
- `scripts/quality_gates.sh` — runs `check_determinism.sh` as the
  `determinism_check` lane whenever determinism-affecting paths change
  (`src/emel/text/generator/**`, `src/emel/kernel/**`, `src/emel/graph/**`,
  `src/emel/memory/**`, `src/emel/tensor/**`, `src/emel/logits/**`,
  `src/emel/sm.hpp`, or the gate itself). Control with
  `EMEL_QUALITY_GATES_DETERMINISM=auto|always|never`; tune with
  `EMEL_DETERMINISM_MODEL`, `EMEL_DETERMINISM_REPEATS`,
  `EMEL_DETERMINISM_TOKENS`.
