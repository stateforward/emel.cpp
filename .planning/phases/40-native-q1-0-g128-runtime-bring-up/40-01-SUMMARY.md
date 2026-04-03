---
phase: 40-native-q1-0-g128-runtime-bring-up
plan: 01
completed: 2026-04-02
status: implemented
---

# Phase 40 Summary

Phase 40 brought the maintained `tests/models/Bonsai-1.7B.gguf` slice onto a truthful EMEL-owned
runtime path by adding real GGUF `Q1_0_g128` support in `src/` on the existing `qwen3` lane.
EMEL's internal packed/prepared pseudo-dtypes were moved out of the upstream GGML/GGUF id range so
raw tensor type `41` now means the real upstream `Q1_0_g128` format without ambiguity.

`src/emel/kernel/detail.hpp` now owns the native `Q1_0_g128` block layout, row dequant helper,
and scalar `Q1_0_g128 x Q8_0` execution used by `mul_mat` and `mul_mat_argmax`. The GGUF loader,
model metadata, and generator tensor-row copy path now all recognize `Q1_0_g128` truthfully, so
the shipped generator backend can initialize and generate through a native quantized path instead
of rejecting the Bonsai tensors or widening into a whole-tensor dequantize-to-f32 fallback.

Focused doctests reproduce and lock the phase contract down at the GGUF loader, kernel, generator
row-copy, and synthetic `qwen3` generator-initialization levels. The quality-gate wrapper timeout
was also raised to `1800s` so the full gate remains green after the benchmark and docs tail,
instead of timing out after benchmark completion.
