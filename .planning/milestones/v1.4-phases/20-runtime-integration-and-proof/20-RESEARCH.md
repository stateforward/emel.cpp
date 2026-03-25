# Phase 20: Runtime Integration And Proof - Research

**Researched:** 2026-03-22
**Domain:** shipped runtime-chain quantized-path attribution and active short-length parity proof
**Confidence:** HIGH

<user_constraints>
## User Constraints

- The user explicitly approved treating `1` and `10` token generation parity as sufficient for
  now, with `100/1000` remaining deferred.
- Runtime proof must not widen public APIs or rewrite actor structure.
- The maintained quantized runtime claim must stay truthful: canonical proof should come from the
  shipped quantized GGUF path, not from the toy f32 generator fixture.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ARCH-02 | The shipped runtime chain adopts the vectorized quantized path without actor rewrites or API widening. | `kernel::any` and `generator::sm` already expose flash attribution additively; q2/q3/q6 attribution can follow the same pattern. |
| PAR-04 | Maintained generation parity proof publishes canonical runtime-chain behavior. | `paritychecker` already loads the real Q2_K GGUF through `generator::sm` and prints runtime attribution, so it is the right proof surface. |
| VER-03 | Regression and kernel tests cover vectorized correctness and deterministic fallback/no-claim behavior. | Existing kernel tests already cover q2/q3/q6 correctness and fallback; Phase 20 only needs runtime-chain attribution proof and no-false-claim coverage. |
</phase_requirements>

## Summary

The runtime-chain seam is already mostly in place. `aarch64::sm` now exposes q2/q3/q6 optimized
and shared dispatch counts, but those metrics stop at the backend seam. `kernel::any` forwards
only flash counters today, and `generator::sm` publishes only flash generation attribution. That
means Phase 20 can stay narrow:

1. extend `kernel::any` with additive q2/q3/q6 optimized/shared accessors
2. extend `generator::sm` with additive generation-time q2/q3/q6 accessors
3. publish those metrics from `paritychecker` on the canonical quantized GGUF runtime path
4. prove that the canonical AArch64 generation path exercises q2/q3/q6 optimized dispatch without
   shared fallback claims
5. prove that the f32 unit generator fixture makes zero quantized optimized/shared claims

The parity gate itself should now align with the user-approved scope. Since the user explicitly
accepted `1` and `10` token parity as sufficient for now, the maintained blocking test can be
narrowed to those lengths while Phase 20 verification continues to record `100/1000` as deferred
debt. That keeps repo gates honest to current user priority without pretending the longer-decode
issue is solved.

## Likely File Changes

| File | Why |
|------|-----|
| `src/emel/kernel/any.hpp` | Add additive q2/q3/q6 optimized/shared forwarding accessors. |
| `src/emel/generator/sm.hpp` | Publish additive generation-time q2/q3/q6 attribution. |
| `tests/generator/lifecycle_tests.cpp` | Prove the f32 fixture does not make false quantized optimized claims. |
| `tools/paritychecker/parity_runner.cpp` | Publish q2/q3/q6 runtime attribution and enforce canonical AArch64 quantized-path proof. |
| `tools/paritychecker/paritychecker_tests.cpp` | Check canonical q2/q3/q6 attribution and narrow enforced parity lengths to `1/10`. |

## Architecture Patterns

### Pattern 1: Additive Wrapper Attribution
Expose backend metrics upward through wrapper accessors only. Do not move ownership or alter
dispatch semantics.

### Pattern 2: Canonical Proof On Real Quantized Runtime
Use the real GGUF loading path in paritychecker for positive proof. Use the toy generator fixture
only for negative/no-claim proof.

### Pattern 3: Explicit Deferred Debt
If user-approved scope is narrower than the roadmap’s full target, keep the narrower gate in code
and record the difference in phase verification.

## Anti-Patterns To Avoid

- Do not claim q2/q3/q6 runtime proof from the f32 generator fixture.
- Do not silently leave `100/1000` in blocking parity tests after the user explicitly deprioritized
  them.
- Do not widen public API signatures or actor structure just to publish attribution.

---
*Phase: 20-runtime-integration-and-proof*
*Research completed: 2026-03-22*
