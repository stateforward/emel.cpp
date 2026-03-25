# Project Milestones: EMEL

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
