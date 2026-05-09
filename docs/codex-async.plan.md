# Codex async ARM inference plan

This is a pre-milestone review of how the `co_sm` coroutine state-machine idea
from `sml.cpp` could be used in EMEL for high-performance ARM inference.

The short version: `co_sm` is useful orchestration plumbing, not a replacement
for native AArch64 kernels. The win is keeping the existing NEON/dotprod/i8mm
kernel paths fed across multiple sequences, cold-load phases, and optional
accelerator submissions. It should not appear inside kernel loops.

## Sources reviewed

- `docs/third_party/sml.md` records the local `co_sm` extension surface:
  `emel::co_sm` accepts a `coroutine_scheduler` policy, defaults to a FIFO
  scheduler, and requires custom schedulers to provide `schedule(fn)` plus
  `guarantees_fifo`, `single_consumer`, and `run_to_completion`.
- `src/emel/sm.hpp` is currently a synchronous wrapper around
  `stateforward::sml::sm`; it normalizes event return values, injects context,
  and provides `sm_any`. There is no tracked `emel::co_sm` wrapper in this tree
  yet.
- `docs/rules/sml.rules.md` still defines the project actor model as
  synchronous RTC with no SML queues, no user mailboxes, no async buffering, and
  no allocation during dispatch.
- `src/emel/kernel/aarch64/**` is the maintained ARM kernel surface. It already
  routes explicit op events through guarded transitions and has optimized
  Q4/Q5/Q6/Q8, packed RHS, matrix x4/x8, i8mm, and flash-attention paths.
- `src/emel/graph/processor/**` has clean phase boundaries
  validate/prepare/alloc/bind/lifecycle/kernel/publish/extract/release. The
  current kernel phase calls a synchronous `run_kernel` callback.
- `src/emel/text/generator/sm.hpp`, `src/emel/logits/sampler/sm.hpp`,
  `src/emel/batch/planner/sm.hpp`, and `src/emel/token/batcher/sm.hpp` already
  model decode, sampling, batch planning, and token batching as explicit SML
  actors with completion-driven phase boundaries.
- `snapshots/bench/benchmarks.txt` already has ARM-relevant baselines for
  generation, AArch64 kernels, flash attention, sampler, tokenizer, memory, and
  batch planner lanes.

## Constraints that shape the design

`co_sm` cannot be adopted as a hidden queue. Before implementation, the milestone
must either:

1. extend `docs/rules/sml.rules.md` with an explicit coroutine actor contract, or
2. keep every `co_sm` use strictly inside a bounded external driver that drains
   continuations to quiescence before returning from the driver tick.

The contract should be:

- A coroutine continuation is not a mailbox message. It is an internal
  continuation owned by one actor or one scheduler tick.
- Every scheduler must be bounded, FIFO, single-consumer per actor, and
  run-to-completion for the chosen RTC boundary.
- Coroutine frames must be allocated before hot dispatch or from fixed
  construction-time pools. Pool exhaustion is a bug, not a heap fallback.
- `co_await` is allowed only at phase boundaries that are already visible in
  the SML graph or introduced as explicit guarded states.
- Runtime behavior selection stays in `guards.hpp` and `sm.hpp`. Coroutine
  bodies and awaitables must not choose dtype, backend, fallback, callback lane,
  error path, or next behavior.
- Kernel code remains kernel-owned. Prefetch, vector loops, i8mm tiles,
  quantized dot products, flash attention, packing, and dequant/quant work stay
  in `src/emel/kernel/**`.

## What `co_sm` can actually buy on ARM

### 1. Continuous decode batching

This is the highest-probability universal win.

ARM LLM decode is usually bandwidth-bound: each token streams a large weight set
from DRAM. The existing AArch64 kernel route already has packed RHS and
matrix-x4/x8 forms. `co_sm` can help the orchestration layer keep enough
sequences ready that those x4/x8 lanes stay full and the same resident weight
tile is amortized across multiple sequences.

Leverage points already present:

- `batch::planner` selects simple/equal/sequential planning modes.
- `token::batcher` normalizes token ids, positions, output masks, and continuity.
- `text::generator` has explicit reset, conditioning, prefill, decode, sample,
  render, and loop states.
- `kernel::aarch64` has explicit guarded routes for packed and matrix batched
  `op_mul_mat` variants.

The milestone should introduce a bounded decode-wavefront driver, not a kernel
rewrite. The driver owns a fixed array of sequence actors, groups ready sequence
lanes by compatible graph/op/dtype/layout, then dispatches existing AArch64
kernel events with full x4/x8 batches when guards allow it.

Success gate: batch sizes 1, 4, 8, and 16 must show decode throughput scaling
close to bandwidth amortization. If x8 is not materially better than x1, stop
and profile kernel layout/KV/cache behavior before adding more async machinery.

### 2. Scalar-side overlap

Sampling, detokenization, formatter work, and some tokenizer variants are scalar
CPU work. The current benchmark snapshot shows tokenizer and sampler lanes are
small compared with one-token generation, but long tokenizer variants can still
be millisecond-scale. `co_sm` can let a decode scheduler advance scalar phases
for one sequence while another sequence is waiting at a graph/kernel boundary.

This is a secondary win. It should be measured after continuous batching, with a
ship gate of at least a small but repeatable latency improvement on
tokenizer-heavy workloads.

### 3. Cold mmap / staged load overlap

The README already describes mmap, read/copy, and staged-read load actors as
owned I/O strategy boundaries. `co_sm` could express first-token cold-load
overlap cleanly:

- Linux ARM: `io_uring`-backed page-range read/fault-in awaitables.
- macOS ARM: bounded `madvise` / residency probing for cold start only.
- Embedded ARM: vendor DMA or accelerator loading only when the platform exposes
  an explicit async completion primitive.

This must never become a steady-state decode path. A hard page fault inside a
kernel is a symptom to measure and fix with residency strategy, not a reason to
put coroutines in the kernel.

### 4. True accelerator async

On CPU-only Apple M-series, Graviton, Ampere, and most Grace CPU-only paths,
there is no user-visible CPU-side async compute engine for NEON/i8mm. `co_sm`
will not make a single synchronous NEON kernel faster.

On Snapdragon, MediaTek, embedded DSP/NPU systems, and accelerator-attached ARM
systems, vendor APIs often provide async graph submission and completion
handles. That is a legitimate `co_await` boundary because the host orchestrator
would otherwise block while another device runs work.

This should be a target-conditional future phase, not the first ARM milestone,
because the current maintained source-backed runtime is AArch64 CPU kernels.

## Target guidance

| Target class | Recommended scheduler | Main experiment | Avoid |
| --- | --- | --- | --- |
| Apple M-series | single-thread inline/FIFO, no cross-cluster hops | continuous batching, tokenizer/sampler overlap | P/E ping-pong, coroutine prefetch |
| Graviton / Ampere / Grace CPU | per-actor single-consumer with possible cross-actor work distribution | continuous batching, Linux cold-load overlap | shared mutable actors, unmeasured multithread scheduler |
| Snapdragon / MediaTek edge | bounded per-actor scheduler plus vendor completion awaitables | NPU/DSP async and continuous batching | generic CPU thread-pool offload as a substitute for kernels |
| Embedded ARM with DMA/DSP | fixed-capacity scheduler tied to vendor completion queue | DMA/accelerator overlap | heap-backed continuations, unbounded request queues |

## Proposed milestone phases

### Phase 0: Pin and codify `co_sm`

- Add or import the actual tracked `co_sm` wrapper/API used by this repository.
- Add a `docs/rules/sml.rules.md` subsection for coroutine actors before any
  code uses them.
- Add compile-only and unit tests for inline scheduler, FIFO scheduler,
  scheduler contract static assertions, fixed-pool allocation, and no-heap
  dispatch.
- Add domain checks preventing coroutine task types from leaking through the C
  ABI or public generator/model contracts.

Acceptance gate: synchronous `emel::sm` behavior remains unchanged, and every
new coroutine test has a deterministic state trace.

### Phase 1: Measurement first

- Add dispatch timeline counters around `process_event`, completion chains, and
  coroutine suspend/resume.
- Add fixed-capacity scheduler metrics: ready depth, continuation count, pool
  high-water mark, immediate-run count, and heap fallback count.
- Snapshot current synchronous baselines for generation, graph processor,
  AArch64 `op_mul_mat`, flash attention, sampler, tokenizer, memory, and batch
  planner lanes.

Acceptance gate: measurement overhead is bounded and repeatable. Do not migrate
actors until the cost of migration can be seen.

### Phase 2: No-op coroutine migration behind an opt-in alias

- Introduce an opt-in `emel::co_sm` wrapper parallel to `emel::sm`; do not
  replace all machines.
- Convert one low-risk, non-kernel actor first with an inline scheduler and no
  actual suspension.
- Measure dispatch overhead against the synchronous baseline.

Recommended first candidates are `batch::planner` or `token::batcher`, because
they are already bounded, benchmarked, and outside kernel hot loops.

Acceptance gate: no-op `co_sm` dispatch is within noise of current `emel::sm`.
If not, keep the synchronous wrapper and stop.

### Phase 3: Decode wavefront scheduler

- Add a bounded decode scheduler that owns a fixed array of per-sequence actors.
- Reuse `batch::planner` and `token::batcher` to form ready sequence groups.
- Group only compatible lanes: same graph phase, same op, same dtype/layout,
  same backend route, same output contract.
- Dispatch the existing `kernel::aarch64` events so existing guards choose
  optimized packed x4/x8 routes.
- Preserve per-sequence deterministic ordering and make stragglers explicit:
  KV growth, sampling, rendering, and tokenizer work can suspend a sequence at a
  phase boundary while other ready sequences advance.

Acceptance gate: tokens/sec improves at useful batch sizes without regressing
batch-1 latency beyond an agreed threshold. A failing gate means the next work is
kernel/KV layout profiling, not more coroutine code.

### Phase 4: Graph processor phase awaitables

- Convert `graph::processor::kernel_step` only if Phase 3 proves the scheduler
  is valuable.
- Replace the synchronous `run_kernel` callback with explicit guarded paths:
  synchronous CPU kernel path stays direct; true async backend path returns an
  awaitable.
- Keep CPU AArch64 kernels synchronous. Only accelerator or OS-backed async
  completion should suspend here.

Acceptance gate: CPU-only AArch64 is not slower, and any async path has a real
external completion source.

### Phase 5: Cold-load overlap

- Add Linux ARM `io_uring` fault-in/read overlap for mmap/staged load paths when
  first-token latency is in scope.
- Add macOS ARM cold-start-only residency hints/probes if measurements justify
  it.
- Keep this outside steady-state decode and outside kernel code.

Acceptance gate: first-token latency improves enough to justify platform code;
steady-state decode is unchanged.

### Phase 6: Target-specific accelerator work

- Add Snapdragon/MediaTek/embedded/Grace+accelerator work only behind
  target-specific routes and explicit user approval.
- Model vendor async submission as a kernel-owned backend event plus an
  awaitable completion boundary.
- Keep reference/parity lanes split: EMEL-owned side must not depend on
  llama.cpp or ggml objects.

Acceptance gate: throughput tracks accelerator slot occupancy, and all
completion/error paths are explicit SML transitions.

## Non-goals

- No `co_await` inside NEON, dotprod, i8mm, flash-attention, packing, or
  quantized dot-product loops.
- No dequantize-to-f32 fallback to make async batching easier.
- No hidden runtime behavior selection in coroutine awaitables or actions.
- No unbounded scheduler queues, heap fallback, or background mailbox semantics.
- No public C ABI exposure of coroutine task or scheduler types.
- No architecture claims without source-backed ARM tests, parity evidence, and
  benchmark evidence.

## Risks

| Risk | Detection | Response |
| --- | --- | --- |
| `co_sm` source/API is not actually tracked in the repo | Phase 0 source review | Pin it before planning implementation |
| Coroutine dispatch overhead exceeds synchronous SML | Phase 2 no-op benchmark | Keep `emel::sm`; use async only at coarse external boundaries |
| Scheduler becomes a mailbox | SML rule audit and deterministic trace tests | Redesign around bounded driver ticks |
| Batch scaling is poor | Phase 3 batch-size benchmark | Profile kernel tile layout, KV cache, and memory bandwidth first |
| Awaitables choose behavior | action/guard branch checks and review | Move choice into guards and `sm.hpp` |
| Heap allocation appears during dispatch | allocator counters and quality gate | Treat as a hard failure |

## Recommendation

Start the next milestone with Phase 0 and Phase 1 only. The most promising
implementation after that is the decode wavefront scheduler, because it directly
uses what EMEL already has: explicit SML phase boundaries, ARM-first kernel
routes, packed x4/x8 AArch64 matmul paths, token batching, and benchmark lanes.

Do not start by refactoring `kernel::aarch64` to be async. Keep kernels
synchronous and allocation-free, then use `co_sm` above them to keep ARM compute
lanes full.
