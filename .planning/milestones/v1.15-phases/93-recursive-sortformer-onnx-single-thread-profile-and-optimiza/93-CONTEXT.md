---
phase: 93
status: executed
created: 2026-04-24
depends_on:
  - 92.6
---

# Phase 93 Context: Recursive Sortformer ONNX Single-Thread Profile And Optimization

## Trigger

Phase `92.6` restored correctness against both independent executable reference lanes:

- PyTorch/NeMo parity reference: `pytorch.nemo.sortformer.v2_1`
- ONNX benchmark reference: `onnx.sortformer.v2_1`
- Shared output: `output_dim=17`, checksum `4249677247906920305`

The milestone still could not close because EMEL had to beat the ONNX Runtime CPU single-thread
benchmark reference on the pinned maintained AMI fixture.

## Constraints

- The EMEL lane must stay EMEL-owned: maintained GGUF loader, real AMI WAV fixture, native
  `src/` runtime, and no ONNX/PyTorch/reference dependency in EMEL execution.
- ONNX remains the benchmark reference lane and must be CPU-only single-thread.
- PyTorch/NeMo remains the parity reference lane and must exact-match before ONNX is used as a
  closeout target.
- Optimization work must land in kernel/runtime-owned code, not tool-only benchmark scaffolding.
- One-time setup/model loading is secondary; steady-state `ns_per_op` is the target.

## Final Evidence

Strict generated record set:

`build/diarization_compare_post_pipeline_pr_feedback`

- EMEL: `1370917625 ns/op`, `15` runs, `output_dim=17`, checksum `4249677247906920305`
- ONNX CPU single-thread: `5900446125 ns/op`, `15` runs, `output_dim=17`,
  checksum `4249677247906920305`, `actual_providers=CPUExecutionProvider`
- PyTorch/NeMo: `11417840125 ns/op`, `1` run, `output_dim=17`,
  checksum `4249677247906920305`

Result: EMEL is `0.232x` ONNX time, so EMEL is faster than the ONNX single-thread benchmark
reference while exact-matching both PyTorch/NeMo and ONNX.
