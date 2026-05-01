# ARM-first Inventory

Last reviewed: 2026-04-29

EMEL is ARM-first. Maintained runtime claims require ARM/AArch64 source-backed execution,
lane-isolated parity evidence, and benchmark evidence. This inventory records non-ARM surfaces
reviewed during ARM-first cleanup and whether removal was clearly safe.

## Maintained ARM Surfaces

- `src/emel/kernel/aarch64/**`
- `tests/kernel/aarch64_tests.cpp`
- `tools/bench/kernel/aarch64_bench.cpp`
- generated architecture docs for `kernel_aarch64`

These surfaces are part of the maintained ARM direction and were preserved.

## Non-ARM Surfaces Reviewed

- `src/emel/kernel/x86_64/**`, `tests/kernel/x86_64_tests.cpp`,
  `tools/bench/kernel/x86_64_bench.cpp`, generated `kernel_x86_64` docs.
- Removed in the ARM-first cleanup: `src/emel/kernel/cuda/**`, `src/emel/kernel/metal/**`,
  `src/emel/kernel/vulkan/**`, and `src/emel/kernel/wasm/**` placeholder actors plus generated
  architecture docs.

## Removal Decision

The CUDA, Metal, Vulkan, and WASM kernel actors were removed. They were placeholder actor surfaces
with no maintained ARM runtime claim, no dedicated benchmark lane, and no implementation beyond
accept/reject plumbing. Keeping them in `kernel::any` made the aggregate kernel domain look broader
than the maintained ARM-first runtime.

The `x86_64` kernel surface was preserved. It has real numeric code, CMake test coverage, benchmark
hooks, and flash-attention comparison paths. Removing it would be a broader behavior and tooling
change, not a clearly safe cleanup, unless a dedicated phase explicitly retires those tests and
benchmark surfaces.
