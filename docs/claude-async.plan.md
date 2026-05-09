# Near-Zero-Overhead Async Inference on ARM with `co_sm`

A pragmatic, measurement-first plan to use `emel::co_sm` (boost/sml/utility/co_sm.hpp)
to drive an async inference engine that is genuinely zero-allocation, RTC-preserving,
and provably faster than a sync baseline across the AArch64 ecosystem — not just one
vendor.

The premise of this plan: **`co_sm` is plumbing, not a kernel optimization.** It cannot
make a single decode step faster. What it can do is keep SIMD/SME/NPU execution units
fed across multiple concurrent requests by overlapping orchestration, I/O, accelerator
dispatch, and KV maintenance with compute. The optimal mechanism varies sharply by
target part; the plan is parameterized accordingly.

## Target matrix

The same `co_sm` source compiles for every target below, but the scheduler, allocator,
and which experiments are worth running differ:

| Target                       | SIMD/Compute            | Accelerator path                  | DMA                | io_uring | Cluster                     | Worthwhile experiments |
|------------------------------|-------------------------|-----------------------------------|--------------------|----------|-----------------------------|------------------------|
| Apple M-series               | NEON+AMX(priv)+SME(M4+) | None user-facing                  | None user-facing   | No       | P/E private L2, costly mig. | Cont. batching, tok overlap |
| AWS Graviton 3 (Neoverse V1) | SVE 256-bit             | None                              | None               | Yes      | Single cluster, SLC         | Cont. batching, io_uring fault-in, multi-thread sched |
| AWS Graviton 4 (Neoverse V2) | SVE2 128-bit + i8mm     | None                              | None               | Yes      | Single cluster              | Same as above |
| Ampere Altra / AmpereOne     | NEON / SVE2             | None                              | None               | Yes      | Single cluster, high core c.| Cont. batching, multi-thread sched |
| NVIDIA Grace                 | SVE2 128-bit            | Hopper via NVLink-C2C             | GPU-side           | Yes      | NUMA, accelerator-attached  | Async accelerator submit (the headline here) |
| Snapdragon X Elite / 8 Gen   | NEON + Hexagon          | **Hexagon NPU (QNN) async**       | **NPU SDK**        | Yes      | DynamIQ, shared L3          | NPU offload via co_await, cont. batching, big.LITTLE viable |
| MediaTek Dimensity / Genio   | NEON + APU              | **APU (NeuroPilot) async**        | **APU SDK**        | Yes      | DynamIQ, shared L3          | NPU offload, cont. batching |
| Embedded (NXP/TI/Renesas)    | NEON + DSP              | Vendor DSP async                  | Real DMA contr.    | Varies   | Varies                      | DMA-overlap, vendor accel offload |

Key takeaways from this matrix:

- **Continuous batching is the only universal win.** Every target benefits.
- **Accelerator/NPU offload via `co_await` is the largest win on edge SoCs and
  accelerator-attached server parts** — those are exactly the configurations where
  the coroutine cost is dominated by accelerator latency.
- **Multi-threaded scheduling is realistic on server ARM** (Graviton, Ampere,
  Grace) where core counts justify the synchronization overhead and DynamIQ-style
  shared L3 keeps cross-core costs sane. It's not realistic on Apple M-series.
- **Hardware DMA / SDK-level async** is real on edge and embedded ARM. My earlier
  "DMA is fiction" claim was specific to Apple Silicon; it does not generalize.

## 1. What we actually buy with coroutines

| Phase                    | Bottleneck             | Async helps?                      |
|--------------------------|------------------------|-----------------------------------|
| Decode (per-token math)  | DRAM bandwidth         | No (kernel-internal)              |
| Prefill (per-token math) | Compute (NEON/SME/AMX) | No (kernel-internal)              |
| Tokenize / detokenize    | Scalar CPU             | Yes (overlap with kernel)         |
| KV-cache copy/grow       | LSU bandwidth          | Marginal (already in-loop on M-class) |
| mmap weight fault-in     | Storage latency        | Yes (overlap first-token latency) |
| Continuous batching      | Scheduler dispatch     | **Yes — the headline win**        |
| HTTP/socket token stream | Syscall + network      | Yes                               |

Only the last four matter. The plan targets continuous batching first, then
tokenizer overlap, then weight fault-in. Everything else is left alone.

## 2. Architectural alignment with `docs/rules/sml.rules.md`

Hard constraints I will not violate (target-independent):

- **Scheduler choice is per-target but always satisfies `strict_ordering_scheduler_contract`.**
  `co_sm`'s static_assert requires `guarantees_fifo && single_consumer && run_to_completion`.
  - On Apple M-series and most edge SoCs: stock `fifo_scheduler` (single-threaded).
  - On server ARM (Graviton, Ampere, Grace) where multi-thread dispatch pays off:
    a custom scheduler that is still **per-actor single-consumer** but allows
    cross-actor work-stealing at a coarser grain. Single-consumer-per-actor is
    what RTC requires; multi-consumer in the global sense is allowed if every
    actor has exactly one logical owner at a time.
- **Pre-sized pools, no heap fallback in steady state.** `pooled_coroutine_allocator`
  sized to `EMEL_MAX_INFLIGHT_SEQUENCES * sizeof(decode_frame)` plus slack;
  exhaustion is a fatal sizing bug, not a runtime fallback.
- **No runtime branching in actions.** Coroutine bodies stay linear; behavior
  selection stays in `guards.hpp` and `sm.hpp` transitions.
- **`co_await` only at phase boundaries that are already RTC boundaries today.**
  Specifically: between cross-actor calls, before/after kernel dispatch, around
  I/O, and around accelerator submission — never inside a kernel loop, never
  inside a guard.
- **Drop-in synchronous path stays free.** Plain `process_event(...)` on `co_sm`
  is one inlined call into `sml::sm::process_event` (line 580). All existing actors
  flip to `co_sm<Model, coroutine_scheduler<inline_scheduler>>` in a no-op migration
  step before any async work begins.

## 3. Measurement harness (Phase 0 — must come first)

You cannot optimize a dispatch overhead you cannot measure. Before any `co_sm`
adoption:

1. **Cycle-accurate dispatch tracer.** Use `mrs x0, cntvct_el0` (cheap, ~1–2 cycles)
   at each `process_event` entry/exit, sm transition, and coroutine
   suspend/resume. Emit to a per-actor ring buffer; flush off the hot path.
2. **Per-phase histogram tool** under `tools/bench/dispatch_timeline/`. Produces
   p50/p95/p99 for tokenize, prefill-step, decode-step, kv-grow, detokenize.
3. **Counters for `co_sm` internals.** Track:
   - `try_run_immediate` hit rate (target: ~100% on single-sequence workloads).
   - `pooled_coroutine_allocator` slot occupancy peak.
   - `fifo_scheduler` queue depth peak.
   - Any heap fallback (must be zero in production runs; assert in debug).
4. **Snapshot baseline.** Capture current sync-dispatch numbers under
   `snapshots/dispatch_baseline_aarch64.json` before changing a line.

Acceptance gate for Phase 0: timeline diffs reproducible to <1% across three runs
on the target part (M-series for the local lane; ARMv8.2+ Linux for CI).

## 4. Phased experiments

### Phase 1 — No-op migration to `co_sm` with `inline_scheduler`

- Change every `using sm = stateforward::sml::sm<model>;` alias touched by
  this plan to `using sm = stateforward::sml::utility::co_sm<model,
  policy::coroutine_scheduler<policy::inline_scheduler>>;`.
- No async API calls anywhere. Pure synchronous `process_event`.
- **Success criterion:** snapshot diff of dispatch timeline within ±1% of the
  Phase 0 baseline. If this regresses, stop and investigate inlining (likely a
  missing `[[gnu::always_inline]]` or LTO config).

### Phase 2 — Continuous batching for decode (the headline experiment)

The actual win. Each in-flight sequence is one `co_sm<decode_sequence>`. A single
scheduler instance owns N of them (N = `max_inflight`, statically sized).

Mechanism:
1. Decode driver actor holds an array of `co_sm<decode_sequence>` (no
   pointer-chasing, all colocated).
2. Each step: scheduler iterates ready sequences, runs each one through one
   decode transition (single matmul over the shared weight tile).
3. Sequences that need KV grow, tokenizer pass, or sampling-with-repetition-
   penalty work suspend at the phase boundary; scheduler advances to the next
   ready one.
4. Weight tile loads from DRAM are amortized across all N sequences in the
   ready set (the actual bandwidth win).

**Why this beats a sync loop:** sync loops force every sequence to hit every
phase per token. Async loops let the scheduler defer per-sequence stragglers
(slow tokenizer, KV growth) so the matmul phase processes the maximum
batchable subset of sequences against a single weight load.

**ARM-specific detail:** the weight tile pulled from DRAM during decode is the
expensive thing. On M3 the L2 holds ~16 MB; a Q4_0 7B model has ~3.5 GB of
weights. We are streaming the entire weight set per token. Going from
batch-1 to batch-8 turns 8× DRAM reads into 1× DRAM read with 8× compute on the
resident tile — bandwidth-amortized at no extra DRAM cost. This is the entire
performance reason async is worth it.

**Acceptance gate:** measured tokens-per-second at batch sizes {1, 4, 8, 16}
shows ≥ (N × 0.7) scaling on memory-bound configs. Below 0.7× efficiency at
N=8 indicates a scheduler stall worth chasing.

### Phase 3 — Tokenizer / detokenizer overlap

- Tokenize coroutine and decode coroutine share the scheduler. Tokenize runs
  while the previous step's detokenize is finishing.
- Overlap window is small (~tens of µs per token at most), but on small models
  with fast decode it's a measurable single-digit-percent win.
- **Acceptance gate:** ≥3% latency reduction on a tokenizer-heavy workload
  (chat with frequent BPE merges) versus Phase 2 baseline. If under 3%, ship
  but don't celebrate.

### Phase 4 — Weight fault-in overlap (first-token latency only)

Two implementations gated on target:

- **Linux ARM (Graviton, Ampere, Grace, edge Linux):** real `io_uring` SQE for
  the layer's page range, completion via CQE, `co_await` natural. This is the
  preferred path everywhere it's available.
- **macOS / non-Linux:** `madvise(MADV_WILLNEED)` followed by a bounded
  `mincore()` poll loop on the page set the coroutine is about to touch.
  Cold-start only; max 100 µs back-off; never on steady-state decode.

Steady-state decode is never on this path on any target.

- **Acceptance gate:** first-token latency on a cold mmap improves by ≥20% on
  a model larger than physical RAM. Smaller wins do not justify the
  complexity; rip it out.

### Phase 5 — Accelerator / NPU async offload (edge and accelerator-attached only)

This is the highest-leverage phase on Snapdragon, MediaTek, Grace+Hopper,
embedded vendor DSPs, and any future ARM target with an async submission
queue. Skipped entirely on Apple M-series, Graviton, Ampere.

- **Mechanism:** the kernel actor wraps the vendor async submission API
  (Hexagon QNN `qnn_graph_execute_async`, NeuroPilot equivalents, CUDA
  streams on Grace, vendor DMA descriptors on embedded) in a custom
  awaitable. The coroutine `co_await`s on completion (event handle / fence
  / IRQ-driven wakeup). The orchestration thread is freed for other
  sequences during the accelerator's compute window.
- **Why coroutines win here:** accelerator latencies are tens to hundreds
  of µs. The orchestrator is doing nothing during those windows in a sync
  design. With async dispatch, the same orchestrator drives N sequences
  against M accelerator slots, hiding the latency behind other work.
- **Multi-thread scheduler is justified here** when the host has more
  cores than accelerator slots, because the orchestrator can prepare
  metadata for the next submission on one core while the previous
  accelerator submission is pending on another.
- **Acceptance gate:** with K accelerator slots and N inflight sequences
  (N > K), measured throughput approaches K × per-slot throughput. If
  throughput is bottlenecked by orchestration, the scheduler is the
  problem.

### Phase 6 — Cross-actor cycles where current code uses callback chains

Several places (`emel/model/loader/**`, request lifecycle in
`emel/runtime/**`) currently express A→B→A patterns as callback or
completion-event gymnastics because the rules forbid recursive
`process_event`. With `fifo_scheduler`, that pattern is legal — B's reply
enqueues onto A's scheduler and drains within the parent RTC.

This is purely a code-clarity win; benchmark first, and only convert call
sites where the new code is provably as fast or faster than the callback
version. **Do not** refactor the kernel actors — those are already optimal.

## 5. Things I am *not* doing (and why) — target-qualified

- **`prfm`-as-coroutine: never, on any target.** Software prefetch is a
  non-blocking instruction that belongs in the kernel inner loop with the
  right cycle distance for the part. It is not async on any ARM. (This is
  distinct from accelerator/DMA async dispatch, which IS legitimate on the
  parts that have it — see Phase 5.)
- **Coroutine-driven kernel inner loops: never, on any target.** The kernel
  is the hot path. `co_await` inside it defeats the purpose.
- **big.LITTLE core dispatch on Apple M-series: no.** P/E migration costs
  ~50–100 µs because of private cluster L2; weight bandwidth is the
  bottleneck, not orchestration. **On DynamIQ ARM SoCs (Snapdragon,
  MediaTek, server ARM): revisit in Phase 5** — shared L3 makes migration
  roughly an order of magnitude cheaper, and the orchestration cost
  becomes meaningful when the accelerator path is hot.
- **DMA engine integration on Apple Silicon and Graviton/Ampere/Grace
  (CPU side): no.** No user-accessible DMA. **On Snapdragon, MediaTek,
  embedded ARM: yes** via the vendor accelerator SDKs (Phase 5).
- **`io_uring` on macOS: not available, fall back to `madvise`+`mincore`.
  On any Linux ARM (server or edge): yes, this is the preferred Phase 4
  implementation.**
- **Multi-threaded scheduler on Apple M-series: no** — orchestration
  overhead dominates and the cluster topology penalizes cross-cluster
  sync. **On server ARM (Graviton, Ampere, Grace) and DynamIQ edge: yes,
  with care** — keep per-actor single-consumer to preserve RTC, allow
  multi-actor parallelism at the scheduler level. Use LSE atomics
  (`CAS`/`LDADD`/`SWP`) for the cross-actor handoff, which are cheap on
  Neoverse V2 / Cortex-X4 / X925.

## 6. AArch64 microarchitecture notes — universal and target-specific

**Universal across AArch64:**
- `pool_slot` uses `alignas(std::max_align_t)` (16 B), matching the AArch64
  ABI stack alignment. No misaligned-load penalty on pool-allocated frames.
- `cntvct_el0` reads ~1–2 cycles; safe to use liberally for tracing.
- HALO frame elision: clang fires it reliably when the awaitable type is
  concrete and the awaiter outlives the awaitee. Plan keeps awaitable
  types concrete to maximize HALO firing across all targets.
- LSE atomics (ARMv8.1+) make `CAS`/`LDADD`/`SWP` cheap; required for
  the multi-thread scheduler path on server ARM. Verified present on
  every server/edge target in the matrix.

**Apple M-series specific:**
- AMX (private) and SME (M4+) execute in dedicated units; the
  scheduling thread can prepare next-sequence metadata in parallel with
  these units across phase boundaries. P/E private L2 makes cross-
  cluster migration expensive (~50–100 µs).

**Server ARM (Graviton 3 V1, Graviton 4 V2, AmpereOne, Grace):**
- SVE/SVE2 register width varies (256-bit on V1, 128-bit on V2 and Grace).
  Kernel layer must size tiles per detected vector length; this is below
  the `co_sm` layer and orthogonal to this plan.
- Shared SLC / system-level cache changes the bandwidth picture vs
  M-series. Continuous batching gains may scale further than M-series
  measurements suggest.
- AArch64 weak memory model becomes load-bearing once the multi-thread
  scheduler is enabled — every enqueue/dequeue needs paired
  `stlr`/`ldar` or LSE-acquire/release semantics. Cost analysis
  required before that scheduler ships.

**DynamIQ edge SoCs (Snapdragon X Elite, Dimensity, Cortex-X+A series):**
- Shared L3 makes cross-cluster migration much cheaper than Apple's
  private-cluster topology. Big.LITTLE-aware scheduling becomes viable
  in Phase 5 specifically because of this.
- NPU/DSP submission queues are interrupt-driven; the natural completion
  primitive is a `eventfd`-style file descriptor (Linux) or vendor
  callback that can drive a custom awaitable.

**NVIDIA Grace + Hopper:**
- NVLink-C2C latency is ~1 µs for small transfers; coarse-grained
  accelerator submission via async streams + `co_await` on
  cudaEventRecord/cudaStreamSynchronize-equivalent is the natural fit.
- NUMA awareness matters; co_sm allocator pools should be per-NUMA-node.

## 7. Risks and what would invalidate the plan

| Risk | Detection | Response |
|------|-----------|----------|
| HALO doesn't fire → frames get pool-allocated → cache pollution | Coroutine pool occupancy > 0 in single-sequence runs | Add `[[gnu::always_inline]]` to awaitable methods; keep awaitable types concrete; if persistent, drop `process_event_async` and stay synchronous |
| `try_run_immediate` hit rate < 95% on single-sequence runs | Phase 0 counter | Investigate re-entrant dispatch; this means actions are violating the no-recursion rule and was being papered over by completion chains |
| Phase 2 batching scaling < 0.7× | Phase 2 acceptance gate | The bottleneck isn't dispatch; it's KV cache layout or matmul tile size. Stop the async work and fix the kernel. |
| Coverage drops below 90% on coroutine paths | `scripts/quality_gates.sh` | Hand-write coverage tests for each `co_await` point; it's not optional |
| Domain leak: `co_sm` types appearing in public C ABI | `scripts/check_domain_boundaries.sh` | Wrap at the C boundary; never let a `bool_task` cross `extern "C"` |

## 8. Order of operations

Universal first three:

1. Phase 0 (measurement) — **blocking on every target**; nothing else starts.
2. Phase 1 (no-op migration to `co_sm` + `inline_scheduler`) — universal.
3. Phase 2 (continuous batching) — universal headline experiment.

Target-conditional remainder:

4. Phase 3 (tokenizer overlap) — universal; only ship if ≥3% gain.
5. Phase 4 (cold-start fault-in) — `io_uring` on Linux ARM, `madvise+mincore`
   on macOS. Only if first-token latency is in scope.
6. Phase 5 (accelerator/NPU async offload) — **only on Snapdragon, MediaTek,
   Grace+Hopper, or embedded ARM with vendor async accelerator path.**
   This is the highest-leverage phase on those targets; on Apple M-series
   and Graviton/Ampere there is no path to enable, skip entirely.
7. Phase 6 (cross-actor cleanup) — opportunistic, universal, never under
   time pressure.

If Phase 2 fails its acceptance gate, Phases 3–6 are suspended and work
returns to kernel-side optimization. Async plumbing is not a substitute
for a faster matmul.

## Conclusion

`co_sm` is the right primitive for an async inference engine across the
AArch64 ecosystem, but the highest-leverage experiment depends on the target:

- **CPU-only ARM (Apple M-series, Graviton, Ampere):** continuous batching is
  the win — N concurrent sequences amortize one streaming weight load.
- **Accelerator-attached ARM (Snapdragon, MediaTek, Grace+Hopper, embedded
  with NPU/DSP):** async accelerator submission via `co_await` is the
  largest win, hiding accelerator latency behind orchestrator work.
- **Linux ARM in general:** `io_uring`-driven cold-start fault-in is real
  and worth doing wherever first-token latency matters.

Targets where `gemini-async.plan.md`'s techniques apply legitimately:
DMA/NPU dispatch on Snapdragon/MediaTek/embedded; some big.LITTLE work on
DynamIQ shared-L3 SoCs. Where they remain wrong on every target: prefetch
as `co_await`, kernel-internal coroutines, and any scheduling design that
violates per-actor RTC.

Build the measurement harness first on every target. Prove the win on the
target you care about. Don't ship plumbing without numbers.
