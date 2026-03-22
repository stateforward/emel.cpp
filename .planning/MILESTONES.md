# Project Milestones: EMEL

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
