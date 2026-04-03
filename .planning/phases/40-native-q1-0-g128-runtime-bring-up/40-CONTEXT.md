---
phase: 40
slug: native-q1-0-g128-runtime-bring-up
created: 2026-04-02
status: complete
---

# Phase 40 Context

## Phase Boundary

Bring the maintained `tests/models/Bonsai-1.7B.gguf` slice onto the shipped EMEL generator path by
adding native `Q1_0_g128` loading, validation, and kernel execution in `src/` without any
whole-tensor dequantize-to-f32 substitution in the hot inference path.

## Implementation Decisions

### True Operand Format
- The maintained Bonsai artifact stores its quantized tensors as GGUF tensor type `41`
  (`Q1_0_g128`), not `TQ1_0`.
- EMEL must therefore support the exact `Q1_0_g128` storage contract: 128 weights per block, one
  fp16 scale, and 128 sign bits.
- Phase 40 ports the Prism/ggml arithmetic for `Q1_0_g128 x Q8_0` scalar execution into
  `src/emel/kernel/detail.hpp` as the first native runtime path.

### Internal Dtype Collision
- EMEL currently uses `41-44` for internal packed pseudo-dtypes, which collides with upstream
- GGUF type `41`.
- Phase 40 must move the project-owned packed/prepared pseudo-dtypes out of the upstream GGML/GGUF
  id range before loader/runtime support for `Q1_0_g128` can be truthful.
- All project-owned packed/prepared dtypes remain EMEL-only and are not exposed as upstream model
  tensor ids.

### Runtime Scope
- Bonsai remains on the existing `qwen3` architecture lane; no new model family is introduced.
- The shipped generator path may initially use the generic native quantized kernel path for
  `Q1_0_g128` tensors, as long as the operand path stays quantized end-to-end and does not
  dequantize the whole tensor to f32.
- Packed/prepared acceleration for `Q1_0_g128` is out of scope for this phase unless required to
  satisfy the existing generator path on supported hosts.

## Existing Code Insights

### Reusable Assets
- `src/emel/gguf/loader/detail.hpp` already centralizes GGUF tensor-type layout validation.
- `src/emel/kernel/detail.hpp` already owns native quantized block structs, row-size accounting,
  dequant helpers, and scalar `mul_mat` / `mul_mat_argmax` implementations.
- `src/emel/generator/detail.hpp` already binds raw quantized matrix tensors into the shipped
  generator backend; if the raw dtype is accepted and the kernel can execute it, no new state
  machine topology is required.

### Confirmed Reference Truth
- Prism fork commit `f5dda7207ed5837f1c83c2f52f851ad9b933d2fd` defines `Q1_0_g128` in
  `ggml-common.h`, `ggml-quants.c`, and `ggml-cpu/quants.c`.
- The maintained Bonsai artifact contains `197` tensors of GGUF type `41`
  (`Q1_0_g128`) and `113` tensors of `F32`.
- Prism's scalar CPU path computes `Q1_0_g128 x Q8_0` by multiplying one block scale with the
  corresponding four `Q8_0` sub-block scales and summing sign-selected int8 values.

### Integration Points
- `src/emel/kernel/events.hpp` and `src/emel/kernel/detail.hpp` must treat `Q1_0_g128` as a real
  upstream tensor dtype and move EMEL-owned packed pseudo-dtypes away from the conflicting ids.
- `src/emel/model/data.cpp` must recognize `Q1_0_g128` as native-quantized and report it
  truthfully in quantized-path audits.
- `tests/kernel/lifecycle_tests.cpp`, `tests/generator/detail_tests.cpp`, and
  `tests/model/loader/lifecycle_tests.cpp` are the right places to reproduce the current failure
  and pin the new contract.
