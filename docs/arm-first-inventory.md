# CPU/backend Inventory

Last reviewed: 2026-04-29

EMEL is CPU-first. Maintained runtime claims require source-backed CPU execution, lane-isolated
parity evidence, and benchmark evidence. This inventory records backend surfaces reviewed during
the earlier backend cleanup and whether removal was clearly safe.

## Maintained CPU Surfaces

- `src/emel/kernel/aarch64/**`
- `tests/kernel/aarch64_tests.cpp`
- `tools/bench/kernel/aarch64_bench.cpp`
- generated architecture docs for `kernel_aarch64`
- `src/emel/kernel/x86_64/**`
- `tests/kernel/x86_64_tests.cpp`
- `tools/bench/kernel/x86_64_bench.cpp`
- generated architecture docs for `kernel_x86_64`

These surfaces are part of the maintained CPU direction and were preserved.

## Device Backend Surfaces Reviewed

- Removed in the earlier backend cleanup: `src/emel/kernel/cuda/**`, `src/emel/kernel/metal/**`,
  `src/emel/kernel/vulkan/**`, and `src/emel/kernel/wasm/**` placeholder actors plus generated
  architecture docs.

## Removal Decision

The CUDA, Metal, Vulkan, and WASM kernel actors were removed. They were placeholder actor surfaces
with no maintained CPU runtime claim, no dedicated benchmark lane, and no implementation beyond
accept/reject plumbing. Keeping them in `kernel::any` made the aggregate kernel domain look broader
than the maintained CPU-first runtime.

The `x86_64` kernel surface was preserved. It has real numeric code, CMake test coverage,
benchmark hooks, and flash-attention comparison paths, so it belongs to the maintained CPU surface.
