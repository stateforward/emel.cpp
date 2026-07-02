# Phase 239 x86_64 Baseline Audit

**Date:** 2026-06-25
**Scope:** Ryzen 9 5950X host contract and pre-kernel x86_64 baseline.

## Host Contract

Phase 239 establishes an EMEL-owned x86_64 host feature contract for the current
CPU feature class:

- Supported: AVX2, FMA, and F16C conversion.
- Explicit no-claim: AVX-512, AVX-VNNI, AMX, BF16 execution, native FP16
  arithmetic, and GPU execution.
- Runtime detection is local CPUID/XGETBV logic in
  `src/emel/kernel/x86_64/context.hpp`, avoiding Zig toolchain link dependency
  on compiler CPU-model symbols.
- Public actor inspection is exposed through `src/emel/kernel/x86_64/sm.hpp`.

## Build Contract

`CMakeLists.txt` now has `EMEL_ENABLE_X86_64_HOST_FEATURES`, matching the
existing AArch64 host-feature switch pattern. On non-cross non-MSVC x86_64
builds, CMake checks and applies only:

- `-mavx2`
- `-mfma`
- `-mf16c`

No AVX-512, AVX-VNNI, AMX, BF16, native-FP16, or GPU flags were added.

The x86 host build also exposed pre-existing non-ARM compile gaps. Phase 239
repaired those without changing runtime contracts:

- AArch64 NEON helper signatures are hidden from non-ARM compilers in
  `src/emel/kernel/aarch64/actions.hpp`.
- ARM-only doctest skip markers now use supported doctest assertions.
- Non-ARM warning-as-error issues in shared generator/diarization/embedding
  helpers are acknowledged with no-op casts.
- `tools/paritychecker/CMakeLists.txt` includes the fetched reference
  implementation's `vendor` directory so reference-side `<nlohmann/json.hpp>`
  resolves.

## Current x86_64 Kernel Baseline

Current `src/emel/kernel/x86_64` support before Phase 240:

- Existing f32 AVX2 execution helpers cover dup/add/sub/mul/div/sqr/sqrt,
  mul_mat, and unary abs/neg/relu where dtype/layout and host support allow it.
- Runtime SIMD choice still flows through x86_64 guards and SML transitions for
  public dispatch; unsupported requests use the existing scalar/shared behavior.
- Flash attention still routes through the shared workspace helper rather than
  an x86_64 AVX2/FMA optimized flash kernel.
- q2_K/q3_K/q6_K AVX2/FMA hot-path kernels do not exist yet.

## NEON/AArch64 Comparison

The AArch64 precedent has a broader maintained optimization surface:

- AArch64 context publishes optimized/shared dispatch counters.
- AArch64 SML transitions distinguish optimized flash attention from shared
  flash behavior.
- AArch64 quantized paths have route counters and packed/vector coverage for
  multiple quantized formats.

The x86_64 path now has the equivalent host/build contract foundation, but not
the flash, quantized-kernel, runtime-parity, or benchmark attribution parity.

## Assigned To Active Follow-On Phases

- Phase 240: AVX2/FMA flash attention kernel and fallback/no-claim behavior.
- Phase 241: q2_K/q3_K x q8_K AVX2/FMA kernels.
- Phase 242: q6_K x q8_K AVX2/FMA kernel and hot-path allocation/operand proof.
- Phase 243: maintained runtime integration and parity proof.
- Phase 244: benchmark attribution and publication truth.
