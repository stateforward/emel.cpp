---
phase: 145
status: passed
superseded-by: 146
---

# Verification 145: Native Quantized Route Evidence Closure

## Verdict

Pass.

## Requirement Mapping

- TEXTGEN-04: Satisfied for source and behavior.
  - The maintained native quantized helper no longer performs hidden dispatch-time packed-q8/q8-k
    fallback selection.
  - The materialized q8 logits choice is explicit in guards and SML transitions.
  - Changed-file scoped quality gate now passes.
- TEXTGEN-07: Satisfied for source and behavior.
  - The missed source regression now targets `matmul_vector_native_quantized(...)` directly.
  - Generator runtime and parity evidence pass.
  - Added branch-focused detail/action/guard tests raise changed-file coverage to 92.7% line and
    50.2% branch.

## Source Checks

- Native helper body is now:

```cpp
return matmul_vector(backend, matrix, input, output);
```

- Materialized q8 logits are selected by:
  - `guard::compute_materialized_scalar_native_quantized_q8_k_supported`
  - `guard::compute_materialized_scalar_native_quantized_kernel_required`
  - prefill equivalents under `text/generator/prefill/guards.hpp`

## Closeout

No Phase 145 blocker remains. The full changed-file scoped quality gate passes after the
`reference_impl` network fetch issue was resolved.
