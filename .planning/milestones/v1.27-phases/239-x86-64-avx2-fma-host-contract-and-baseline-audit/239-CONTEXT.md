# Phase 239: x86_64 AVX2/FMA Host Contract and Baseline Audit - Context

**Gathered:** 2026-06-25
**Status:** Source-complete; benchmark snapshot update awaiting explicit approval
**Mode:** Auto-generated (autonomous infrastructure phase)

<domain>
## Phase Boundary

Define and prove the x86_64 Ryzen host feature contract before adding new flash
or quantized kernels. This phase may add feature detection, public actor
accessors, host-tuned build flags, tests, and a source-backed audit artifact. It
must not implement the AVX2/FMA flash kernel or the q2_K/q3_K/q6_K hot-path
kernels; those are Phase 240-242 work.

</domain>

<decisions>
## Implementation Decisions

### Host Feature Contract
- **D-01:** Treat this processor as x86_64 AVX2 + FMA with F16C conversion
  support.
- **D-02:** Publish AVX2, FMA, and F16C as explicit supported feature booleans
  on the x86_64 kernel actor context/surface.
- **D-03:** Publish unsupported feature families as explicit no-claim booleans:
  AVX-512, AVX-VNNI, AMX, BF16, and native FP16.
- **D-04:** Keep no-claim feature families disabled even if a future host can
  report them; adding those paths requires a separate milestone contract.

### Build Contract
- **D-05:** Add an x86_64 host-feature build option analogous to the existing
  AArch64 host-feature option.
- **D-06:** Use compiler-checked AVX2/FMA/F16C flags only; do not add AVX-512,
  VNNI, AMX, BF16, native FP16, or GPU flags.
- **D-07:** Preserve portable builds by keeping the new option configurable and
  by applying flags only for non-cross x86_64 builds when supported.

### Audit Scope
- **D-08:** Capture the current x86_64 kernel state against the NEON precedent:
  existing f32 AVX2 paths are present, but flash/quantized parity is not yet at
  the AArch64 standard.
- **D-09:** The audit is evidence for Phase 239 only; it must not claim Phase
  240-244 runtime or benchmark completion.

### the agent's Discretion
- Use the smallest code shape that keeps the feature contract inspectable from
  tests and follow-on phases.
- Prefer focused x86_64 kernel tests and CMake/source scans over broad quality
  gates until implementation files are touched.

</decisions>

<canonical_refs>
## Canonical References

### Current milestone
- `.planning/REQUIREMENTS.md` - `X86-01` and `X86-02` requirements.
- `.planning/ROADMAP.md` - Phase 239 goal and success criteria.
- `.planning/STATE.md` - v1.27 host feature scope and no-claim constraints.

### Existing x86_64 surface
- `src/emel/kernel/x86_64/context.hpp` - current AVX2 detection and context.
- `src/emel/kernel/x86_64/actions.hpp` - current f32 AVX2 execution helpers.
- `src/emel/kernel/x86_64/guards.hpp` - current x86_64 SIMD route guards.
- `src/emel/kernel/x86_64/sm.hpp` - public x86_64 actor wrapper surface.
- `tests/kernel/x86_64_tests.cpp` - existing x86_64 actor/kernel coverage.

### NEON precedent
- `src/emel/kernel/aarch64/context.hpp` - feature/counter precedent.
- `src/emel/kernel/aarch64/sm.hpp` - public counter/accessor precedent.
- `.planning/milestones/v1.3-ROADMAP.md` - ARM flash optimization pattern.
- `.planning/milestones/v1.4-ROADMAP.md` - vectorized quantized kernel pattern.
- `.planning/milestones/v1.5-ROADMAP.md` - full ARM quantized path proof.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CMakeLists.txt` already has `EMEL_ENABLE_AARCH64_HOST_FEATURES` and
  compiler-checked host flags.
- `src/emel/kernel/x86_64/context.hpp` already detects AVX2 with
  `__builtin_cpu_supports` on GCC/Clang x86_64.
- `src/emel/kernel/x86_64/actions.hpp` already has AVX2 target attributes and
  f32 SIMD helpers for dup/add/sub/mul/div/sqr/sqrt/mul_mat/unary.
- `tests/kernel/x86_64_tests.cpp` already has forced-SIMD and fallback tests.

### Established Patterns
- x86_64 behavior selection belongs in guards/transitions; action/detail code
  only executes an already selected path.
- AArch64 publishes runtime counters/accessors from `sm.hpp`; x86_64 can use the
  same public actor-wrapper style for feature contract inspection.
- Planning artifacts must not be used as proof of runtime support; tests and
  source scans must back every closeout claim.

### Integration Points
- Add feature contract state in `src/emel/kernel/x86_64/context.hpp`.
- Add public wrapper accessors in `src/emel/kernel/x86_64/sm.hpp`.
- Add host-feature compile flag support in `CMakeLists.txt`.
- Add focused x86_64 tests in `tests/kernel/x86_64_tests.cpp`.
- Add Phase 239 audit artifact under this phase directory.

</code_context>

<specifics>
## Specific Ideas

- The current host inspection reports AMD Ryzen 9 5950X with AVX2, FMA, and
  F16C flags.
- The milestone wording must stay honest: this phase starts x86_64 support work;
  it does not complete flash, quantized kernels, runtime parity, or benchmark
  publication. Those items remain active v1.27 scope with concrete follow-on
  phases and phase-owned acceptance criteria.

</specifics>

<active_follow_on_scope>
## Active Follow-On Scope

- AVX2/FMA flash-attention implementation - Phase 240.
- AVX2/FMA q2_K/q3_K kernels - Phase 241.
- AVX2/FMA q6_K and hot-path operand-fidelity proof - Phase 242.
- Runtime parity and benchmark publication - Phases 243-244.

</active_follow_on_scope>

---

*Phase: 239-x86-64-avx2-fma-host-contract-and-baseline-audit*
*Context gathered: 2026-06-25*
