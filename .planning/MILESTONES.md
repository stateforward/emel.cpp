# Project Milestones: EMEL

## v1.11 TE-75M GGUF Trimodal Embedding Runtime (Shipped: 2026-04-15)

**Phases completed:** 3 phases, 3 plans, 0 tasks

**Key accomplishments:**

- (none recorded)

---

## v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice (Shipped: 2026-04-02)

**Phases completed:** 8 phases, 9 plans, 0 tasks

**Key accomplishments:**

- Documented one official maintained `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture with executable
  metadata truth anchored on `lfm2` and `128000` context.

- Added explicit `lfm2` model and runtime support for one maintained Liquid ARM slice through the
  shipped generator path.

- Proved maintained Liquid parity against `llama.cpp` while preserving additive maintained Qwen
  coverage.

- Published one maintained Liquid benchmark/docs path aligned with the parity-backed fixture and
  contract.

- Reconstructed missing closeout artifacts and validation coverage so the milestone could pass
  audit and archive cleanly.

**What's next:** Start the next milestone. The likely follow-on work is broader Liquid coverage,
richer Liquid request support, or Liquid performance work on top of the shipped maintained slice.

---

## v1.8 Truthful Qwen3 E2E Embedded Size (Shipped: 2026-04-02)

**Delivered:** EMEL now publishes one truthful maintained Qwen3-0.6B end-to-end executable-size
comparison between an EMEL-owned runner and a matched `llama.cpp` reference executable, backed by
runtime smoke proof and generated README evidence.

**Phases completed:** 6 phases, 8 plans, 0 tasks

**Key accomplishments:**

- Locked one exact maintained workload boundary on `tests/models/Qwen3-0.6B-Q8_0.gguf`,
  structured `hello`, and first-token generation.

- Corrected the EMEL probe into a truthful final executable measurement and removed redundant
  fallback-vocab bloat that had inflated the binary to 56 MB.

- Kept the published comparator set narrow to EMEL and one matched `llama.cpp` reference
  executable with shared smoke proof.

- Refreshed the stored snapshot and generated README to publish the corrected executable-size
  values: `4,073,016` raw bytes for EMEL versus `3,334,264` for the reference row.

- Backfilled the missing v1.8 proof chain and closed the milestone with a passing `8/8`
  requirement audit.

---

## v1.7 Generator Prefill Submachine Decomposition (Shipped: 2026-03-30)

**Delivered:** The generator now ships an explicit generator-owned `prefill` child machine with
request-scoped prefill contracts, a materially smaller top-level generator surface, and maintained
Llama/Qwen proof preserved across the refactor.

**Phases completed:** 3 phases, 6 plans, 0 tasks

**Key accomplishments:**

- Collapsed the repeated top-level prefill routing matrix into explicit request-scoped prefill
  compute contracts.

- Extracted `src/emel/generator/prefill` as a generator-domain child machine with its own typed
  `run` event and explicit contract/result states.

- Kept prefill orchestration data on typed runtime/internal events instead of generator context
  phase flags.

- Reduced the parent generator surface materially and published the split `generator_prefill`
  architecture docs.

- Re-ran maintained generator, paritychecker, and compare proof on the extracted prefill boundary,
  with unrelated broad benchmark drift explicitly waived for milestone closeout.

**What's next:** Decide whether to continue Issue `#41` with decode extraction or revisit deeper
generator-family decomposition after the prefill pattern.

---

## v1.6 Qwen3-0.6B Parity And Benchmark (Shipped: 2026-03-30)

**Delivered:** EMEL now ships one truthful canonical `Qwen3-0.6B-Q8_0.gguf` maintained slice with
an explicit GGUF-derived formatter contract, native `src/emel` `q8_0` runtime support, stored
generation parity, and benchmark publication aligned to the same operator-facing Qwen workflow.

**Phases completed:** 5 phases, 12 plans, 6 tasks

**Key accomplishments:**

- Locked one official canonical Qwen3 fixture and primary-template-only formatter contract on the
  maintained paritychecker and benchmark surfaces

- Added native `q8_0` runtime support for the canonical Qwen blocker tensors in `src/emel`
- Brought the shipped generator path up on the canonical `qwen3` slice without broad family claims
- Proved maintained stored-baseline parity across `1/10/100/1000` while keeping the prior Llama
  anchor protected

- Published one truthful canonical Qwen benchmark compare, snapshot, and docs path with explicit
  formatter/runtime evidence

**What's next:** Define the next milestone before broadening Qwen scope, widening request surfaces,
or hardening benchmark-gate policy.

---

## v1.5 Full ARM Quantized Path (Shipped: 2026-03-27)

**Delivered:** The canonical CPU-hosted Llama-68M ARM slice now ships an explicit maintained
quantized-path contract, zero supported disallowed fallback, and restored checked-in flash
attribution/publication across paritychecker, compare snapshots, and generated benchmark docs.

**Phases completed:** 5 phases, 10 plans, 0 tasks

**Key accomplishments:**

- The canonical ARM slice now has a shared stage-by-stage quantized-path audit
- Unsupported quantized branches now publish explicit no-claim behavior
- The shipped generator runtime, paritychecker, and regression surfaces now prove the canonical
  `8/4/0/0` runtime contract with zero supported disallowed fallback

- Benchmark compare output, stored snapshots, and generated docs now publish the same runtime
  contract without overstating approved dense-f32-by-contract seams

- Canonical flash-attention dispatch and checked-in benchmark publication were restored together so
  maintained live proof and stored evidence match again

**What's next:** Define the next milestone before widening scope beyond the canonical CPU-hosted
Llama-68M ARM slice or changing benchmark-gate policy.

---

## v1.4 Full Vectorized Quantized Kernels (Shipped: 2026-03-25)

**Delivered:** The canonical CPU-hosted Llama-68M ARM slice now ships EMEL-owned vectorized
`q2_K/q3_K/q6_K x q8_K` kernels, maintained runtime attribution, full `1/10/100/1000` parity
proof, and refreshed benchmark publication against the preserved v1.3 scalar baseline.

**Phases completed:** 5 phases, 11 plans, 0 tasks

**Key accomplishments:**

- Replaced the maintained scalar `q2_K`, `q3_K`, and `q6_K` row helpers with EMEL-owned
  vectorized AArch64 kernels on the canonical operand path.

- Closed the maintained quantized hot-path contract with alloc-free q2/q3/q6 dispatch and no
  dequantize-to-f32 fallback.

- Exposed shipped q2/q3/q6 optimized-versus-shared runtime attribution without widening the
  actor or API surface.

- Restored maintained parity across `1`, `10`, `100`, and `1000` tokens on the canonical ARM
  workload.

- Refreshed maintained benchmark compare output and docs with quantized attribution and preserved
  v1.3 baseline context.

**What's next:** Define the next milestone before broadening beyond the canonical CPU-hosted
Llama-68M ARM slice or widening benchmark-gate policy.

---

## v1.3 ARM Flash Optimizations (Shipped: 2026-03-22)

**Delivered:** Optimized AArch64 flash execution now ships on the canonical Llama-68M ARM slice,
with parity and benchmark surfaces publishing optimized-vs-shared attribution and maintained docs
preserving the prior ARM baseline while showing a measured short-case improvement.

**Phases completed:** 3 phases, 7 plans, 0 tasks

**Key accomplishments:**

- The maintained AArch64 flash request now has a native backend execution path.
- Phase 14 closed kernel-local proof for correctness, scratch reuse, and zero shared fallback.
- The shipped generator seam now reports optimized-vs-shared flash path selection.
- Paritychecker and benchmark compare output now publish optimized-vs-shared ARM flash
  attribution on the maintained workload.

- Maintained benchmark publication now preserves the prior ARM baseline and documents a `1.140x`
  short-case speedup.

**What's next:** Define the next milestone before broadening scope beyond the canonical
CPU-hosted Llama-68M ARM slice.

---

## v1.2 Flash Attention (Shipped: 2026-03-22)

**Delivered:** The canonical Llama-68M generation slice now runs through an EMEL-owned
flash-attention path, paritychecker proves it on the normal surface, the shipped runtime is
hard-cut over to `emel::tensor::sm`, and benchmark docs publish maintained flash evidence with a
preserved pre-flash baseline.

**Phases completed:** 5 phases, 13 plans, 13 tasks

**Key accomplishments:**

- Added a real EMEL-owned flash-attention kernel path with backend-owned workspace reuse.
- Adopted flash attention in the shipped generator runtime with deterministic unsupported-request
  failure proof.

- Made paritychecker fetch and publish upstream reference identity while proving flash execution on
  both maintained workloads.

- Hard-cut graph and generator tensor lifecycle orchestration over to `emel::tensor::sm` with an
  alloc-free dispatch proof.

- Published canonical benchmark proof comments, a preserved pre-flash artifact, and generated docs
  showing a `9.126x` short-case improvement over the prior EMEL baseline.

**What's next:** Define the next milestone before widening scope beyond the canonical
CPU-hosted Llama-68M slice.

---

## v1.1 EMEL Llama-68M Generation Benchmark (Shipped: 2026-03-11)

**Delivered:** One truthful canonical Llama-68M generation benchmark in `tools/bench`, using a
shared native EMEL decode backend and the existing compare, snapshot, and docsgen workflow.

**Phases completed:** 4 phases, 10 plans

**Key accomplishments:**

- Added the canonical generation benchmark case for the shipped Llama-68M slice.
- Replaced the circular reference-backed decode seam with a shared native EMEL backend.
- Published stable EMEL-vs-`llama.cpp` compare output through the maintained compare surface.
- Integrated benchmark snapshots and generated benchmark docs into the existing operator workflow.

**What's next:** Add EMEL-owned flash attention to the same canonical generation slice.

---

## v1.0 EMEL Llama-68M Generation Slice (Shipped: 2026-03-08)

**Delivered:** The first parity-checked canonical Llama-68M generation slice through the existing
EMEL runtime and `tools/paritychecker`.

**Phases completed:** 7 phases, 15 plans

**Key accomplishments:**

- Implemented the real GGUF/model-loading path for the shipped canonical fixture.
- Wired bounded generation end to end through the existing EMEL runtime.
- Added parity-oriented subprocess success and failure coverage for the maintained workload.
- Established the narrow paritychecker-first acceptance boundary used by later milestones.

**What's next:** Build a truthful benchmark surface on top of the proven generation slice.

---
