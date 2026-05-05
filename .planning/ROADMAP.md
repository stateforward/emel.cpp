# Roadmap: EMEL

## Milestones

- ✅ **v1.0 EMEL Llama-68M Generation Slice** — shipped 2026-03-08
- ✅ **v1.1 EMEL Llama-68M Generation Benchmark** — shipped 2026-03-11
- ✅ **v1.2 Flash Attention** — shipped 2026-03-22
- ✅ **v1.3 ARM Flash Optimizations** — shipped 2026-03-22
- ✅ **v1.4 Full Vectorized Quantized Kernels** — shipped 2026-03-25
- ✅ **v1.5 Full ARM Quantized Path** — shipped 2026-03-27
- ✅ **v1.6 Qwen3-0.6B Parity And Benchmark** — shipped 2026-03-30
- ✅ **v1.7 Generator Prefill Submachine Decomposition** — shipped 2026-03-30
- ✅ **v1.8 Truthful Qwen3 E2E Embedded Size** — shipped 2026-04-02
- ✅ **v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice** — shipped 2026-04-02
- ✅ **v1.11 TE-75M GGUF Trimodal Embedding Runtime** — shipped 2026-04-15
- ✅ **v1.12 Pluggable Reference Parity Bench Architecture** — shipped 2026-04-18
- ✅ **v1.13 Pluggable Generative Parity Bench** — shipped 2026-04-21
- ✅ **v1.14 Benchmark Variant Organization** — shipped 2026-04-21
- ✅ **v1.15 ARM Sortformer Diarization GGUF Slice** — shipped 2026-04-25
- ✅ **v1.16 ARM Whisper GGUF Parity And Performance** — shipped 2026-04-28
- ✅ **v1.17 Text Generator Domain Alignment** — shipped 2026-04-30
- ✅ **v1.18 Parity Tool Boundary Refactor** — shipped 2026-05-01
- ✅ **v1.19 Benchmark Tool Pluggable Runner Refactor** — shipped 2026-05-01
- ✅ **v1.20 SML Dependency And Namespace Migration** — shipped 2026-05-02
- ✅ **v1.21 Quality Gate Selective Runner Optimization** — shipped 2026-05-02
- ✅ **v1.22 Weight Loading Ownership Cutover** — shipped 2026-05-03
- ✅ **v1.23 I/O Loading Strategy Boundary** — shipped 2026-05-04
- ✅ **v1.24 I/O Mmap Loading Strategy** — shipped 2026-05-04 (Phases 204-211)

## Phases

<details>
<summary>✅ v1.24 I/O Mmap Loading Strategy (Phases 204-211) — SHIPPED 2026-05-04</summary>

- [x] Phase 204: Mmap Strategy Component Boundary (1/1 plans) — completed 2026-05-04
- [x] Phase 205: Mmap Validation and Platform Gating (1/1 plans) — completed 2026-05-04
- [x] Phase 206: Mapped Descriptor, Errors, and Lifetime (1/1 plans) — completed 2026-05-04
- [x] Phase 207: Tensor-Owned Mmap Integration (1/1 plans) — completed 2026-05-04
- [x] Phase 208: Public Runtime and Evidence Surfaces (1/1 plans) — completed 2026-05-04
- [x] Phase 209: Behavior Tests and Scope Guardrails (1/1 plans) — completed 2026-05-04
- [x] Phase 210: Publication and Maintained Artifact Updates (1/1 plans) — completed 2026-05-04
- [x] Phase 211: Phase Verification Artifact Backfill (1/1 plans) — completed 2026-05-04 (gap closure)

Archive:
- `.planning/milestones/v1.24-ROADMAP.md`
- `.planning/milestones/v1.24-REQUIREMENTS.md`
- `.planning/milestones/v1.24-MILESTONE-AUDIT.md`
- `.planning/milestones/v1.24-phases/{204..210}-*` (Phase 211 backfill artifacts live alongside their parent phase dirs)

</details>

<details>
<summary>✅ v1.23 I/O Loading Strategy Boundary (Phases 197-203) — SHIPPED 2026-05-04</summary>

Archive:
- `.planning/milestones/v1.23-ROADMAP.md`
- `.planning/milestones/v1.23-REQUIREMENTS.md`
- `.planning/milestones/v1.23-MILESTONE-AUDIT.md`
- `.planning/milestones/v1.23-phases/`

</details>

### 📋 Next Milestone

Define the next milestone (v1.25 or higher) via `$gsd-new-milestone`. Concrete read/copy,
async, and device loading strategies remain deferred follow-on work below the
`emel/io` boundary.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 204. Mmap Strategy Component Boundary | v1.24 | 1/1 | Complete | 2026-05-04 |
| 205. Mmap Validation and Platform Gating | v1.24 | 1/1 | Complete | 2026-05-04 |
| 206. Mapped Descriptor, Errors, and Lifetime | v1.24 | 1/1 | Complete | 2026-05-04 |
| 207. Tensor-Owned Mmap Integration | v1.24 | 1/1 | Complete | 2026-05-04 |
| 208. Public Runtime and Evidence Surfaces | v1.24 | 1/1 | Complete | 2026-05-04 |
| 209. Behavior Tests and Scope Guardrails | v1.24 | 1/1 | Complete | 2026-05-04 |
| 210. Publication and Maintained Artifact Updates | v1.24 | 1/1 | Complete | 2026-05-04 |
| 211. Phase Verification Artifact Backfill | v1.24 | 1/1 | Complete | 2026-05-04 |
