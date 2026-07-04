# Roadmap: EMEL

## Milestones

- [x] **v1.0 EMEL Llama-68M Generation Slice** - shipped 2026-03-08
- [x] **v1.1 EMEL Llama-68M Generation Benchmark** - shipped 2026-03-11
- [x] **v1.2 Flash Attention** - shipped 2026-03-22
- [x] **v1.3 ARM Flash Optimizations** - shipped 2026-03-22
- [x] **v1.4 Full Vectorized Quantized Kernels** - shipped 2026-03-25
- [x] **v1.5 Full ARM Quantized Path** - shipped 2026-03-27
- [x] **v1.6 Qwen3-0.6B Parity And Benchmark** - shipped 2026-03-30
- [x] **v1.7 Generator Prefill Submachine Decomposition** - shipped 2026-03-30
- [x] **v1.8 Truthful Qwen3 E2E Embedded Size** - shipped 2026-04-02
- [x] **v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice** - shipped 2026-04-02
- [x] **v1.11 TE-75M GGUF Trimodal Embedding Runtime** - shipped 2026-04-15
- [x] **v1.12 Pluggable Reference Parity Bench Architecture** - shipped 2026-04-18
- [x] **v1.13 Pluggable Generative Parity Bench** - shipped 2026-04-21
- [x] **v1.14 Benchmark Variant Organization** - shipped 2026-04-21
- [x] **v1.15 ARM Sortformer Diarization GGUF Slice** - shipped 2026-04-25
- [x] **v1.16 ARM Whisper GGUF Parity And Performance** - shipped 2026-04-28
- [x] **v1.17 Text Generator Domain Alignment** - shipped 2026-04-30
- [x] **v1.18 Parity Tool Boundary Refactor** - shipped 2026-05-01
- [x] **v1.19 Benchmark Tool Pluggable Runner Refactor** - shipped 2026-05-01
- [x] **v1.20 SML Dependency And Namespace Migration** - shipped 2026-05-02
- [x] **v1.21 Quality Gate Selective Runner Optimization** - shipped 2026-05-02
- [x] **v1.22 Weight Loading Ownership Cutover** - shipped 2026-05-03
- [x] **v1.23 I/O Loading Strategy Boundary** - shipped 2026-05-04
- [x] **v1.24 I/O Mmap Loading Strategy** - shipped 2026-05-04
- [x] **v1.25 I/O Read Loading Strategy** - shipped 2026-05-06
- [x] **v1.26 I/O Staged Read Loading Strategy** - completed 2026-05-08
- [x] **v1.27 Ryzen AVX2/FMA Kernel Support** - shipped 2026-06-25

## Current Milestone

No active milestone is open.

## Recently Shipped

### v1.27 Ryzen AVX2/FMA Kernel Support

**Shipped:** 2026-06-25
**Archive:** `.planning/milestones/v1.27-ROADMAP.md`
**Requirements:** `.planning/milestones/v1.27-REQUIREMENTS.md`
**Audit:** `.planning/milestones/v1.27-MILESTONE-AUDIT.md`

Delivered native x86_64 AVX2/FMA support for the AMD Ryzen 9 5950X maintained
runtime slice: host feature contract, optimized flash attention, q2_K/q3_K/q6_K
x q8_K kernels, maintained generator parity attribution, and truthful
`kernel_x86_64` benchmark publication. The source-backed audit passed after
repairing the optimized benchmark attribution gap and removing the x86_64 unary
SML rule debt.

Next step: run `$gsd-new-milestone` to define the next milestone.
