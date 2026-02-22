# kernel/device/cpu/any architecture design (draft)

this document defines kernel/device/cpu/any. it is the cpu-level `sm_any` dispatcher
that selects a concrete architecture variant for cpu kernel scheduling.

## role
- dispatch graph scheduling to the architecture variant matching the host cpu.
- own thread pool and scratch buffer lifecycle (deferred to threading iteration).

## variants
- `kernel/device/cpu/x86_64` — x86-64 (ISA tiers: scalar, AVX2, AVX-512 selected internally per-op).
- `kernel/device/cpu/aarch64` — ARM 64-bit (ISA tiers: scalar, NEON, AMX selected internally per-op).
- `kernel/device/cpu/wasm` — WebAssembly (scalar, SIMD128 selected internally per-op).

## events (draft)
- `event::schedule` inputs: bound `graph`, cpu execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- mirrors the active variant's state model via `sm_any` dispatch.

## responsibilities
- detect host architecture at construction and select variant.
- forward `event::schedule` to the active variant.
- own thread pool handle and scratch buffer; inject into variants by reference.
- each variant internally selects the best ISA tier per-op via function pointer table
  populated at construction from runtime feature detection (e.g. CPUID, getauxval).
