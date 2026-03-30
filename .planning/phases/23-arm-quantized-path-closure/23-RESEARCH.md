# Phase 23: ARM Quantized Path Closure - Research

**Researched:** 2026-03-25
**Domain:** Supported canonical ARM quantized-path closure
**Confidence:** HIGH

## Summary

I do not find a remaining supported canonical disallowed-fallback gap in the shipped `src/`
runtime path. The current maintained Llama-68M ARM slice still has exactly two widening seams in
generator code: token embedding row materialization through `copy_tensor_row(...)` and norm-vector
materialization through `dequantize_tensor_vector(...)`. Phase 22 already classified those seams as
`approved_dense_f32_by_contract`, not disallowed fallback.

The supported q2/q3/q6 learned-matrix path is still genuinely native quantized. In
`src/emel/generator/detail.hpp`, every learned matrix multiply goes through `matmul_vector(...)`,
which dispatches `kernel::event::op_mul_mat`. In `src/emel/kernel/detail.hpp::can_run_mul_mat(...)`,
the supported quantized path requires quantized `src0`, dense `f32` `src1`, dense `f32` `dst`,
contiguous rhs/dst, and a bounded `QK_K` block shape. In
`src/emel/kernel/aarch64/actions.hpp::execute_neon_mul_mat(...)`, that path repacks dense rhs
activations into bounded `q8_K` scratch and executes q2/q3/q6 row-dot kernels directly. That is
the approved shipped contract, not a whole-row dequantize-to-f32 fallback.

So Phase 23 should not invent a fake runtime closure bug. It should instead close `PATH-01` by
hardening the shipped generator/runtime surface around the zero-gap truth: supported canonical
requests currently have zero `disallowed_fallback` audited stages, and any unsupported branch
continues to surface as explicit no-claim instead of silently widening.

## Evidence

- `src/emel/generator/detail.hpp:251-307`
  Token embedding and norm-vector materialization are the only visible row/vector dequant seams.
- `src/emel/generator/detail.hpp:535-552`
  Learned matrix multiplies dispatch through `op_mul_mat` instead of row-wise dequant helpers.
- `src/emel/generator/detail.hpp:1021-1054`
  Backend initialization binds quantized matrices directly while only dequantizing norms.
- `src/emel/kernel/detail.hpp:1606-1633`
  `can_run_mul_mat(...)` accepts either all-f32 or quantized-lhs-plus-f32-rhs, with no
  whole-matrix dequant fallback branch for supported q2/q3/q6 requests.
- `src/emel/kernel/aarch64/actions.hpp:873-928`
  Supported quantized AArch64 execution repacks rhs activations to `q8_K` scratch and runs q2/q3/q6
  row-dot kernels directly.
- `src/emel/model/data.cpp:398-444`
  Phase 22's audit helper currently reports zero `disallowed_fallback` stages on the canonical
  supported model and marks unsupported dtypes as `explicit_no_claim`.

## Recommended Plan Split

### 23-01: Runtime Contract Closure Surface

- Move the Phase 22 audit into the shipped generator/backend surface so the runtime wrapper can
  report zero `disallowed_fallback` stages directly, not only via tool-local recomputation.
- Expose additive generator accessors for audited stage counts or contract-state queries.
- Keep behavior unchanged for the current canonical model because the supported disallowed count is
  already zero.

### 23-02: Runtime Proof Of Zero Remaining Disallowed Fallback

- Add generator-focused proof that the canonical supported runtime path reports zero
  `disallowed_fallback` stages while preserving the existing approved dense-f32-by-contract seams.
- Keep unsupported-stage proof as explicit no-claim rather than converting it into a synthetic
  disallowed-fallback path.
- Leave paritychecker hard-fail regression across `1/10/100/1000` to Phase 24, where the roadmap
  already places maintained proof expansion.

## User Approval Check

No user approval is needed for the recommended Phase 23 work if it stays additive in helpers,
runtime accessors, and tests. I do **not** recommend changing SML transition tables or actor
ownership here. If someone wanted to rewrite machine structure to enforce the contract, that would
require explicit user approval under `AGENTS.md`.
