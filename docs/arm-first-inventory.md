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

## Apple GPU Surface

- `src/emel/kernel/metal/**` (SML actor + embedded MSL kernels)
- `tests/kernel/metal_tests.cpp` (op parity vs the CPU backends)
- compiled only when the Metal framework is present (Apple hosts); on other
  hosts the actor builds as a portable stub whose guards reject every dispatch

The Metal actor serves the Mimi codec op set (`op_mul_mat` f32/f16/q8_0,
`op_add`, `op_unary`, `op_im2col`, `op_conv_transpose_1d`, `op_get_rows`) and
is opt-in via `kernel::any` `set_kind(kernel_kind::metal)`; the host default
stays CPU-first so CPU parity and benchmark lanes never change operand
pipelines silently. No CPU benchmark claims depend on it.

## Device Backend Surfaces Reviewed

- Removed in the earlier backend cleanup: `src/emel/kernel/cuda/**`,
  `src/emel/kernel/vulkan/**`, and `src/emel/kernel/wasm/**` placeholder
  actors plus generated architecture docs. (The Metal placeholder was
  replaced by the maintained surface above.)

## Removal Decision

The CUDA, Metal, Vulkan, and WASM kernel actors were removed. They were placeholder actor surfaces
with no maintained CPU runtime claim, no dedicated benchmark lane, and no implementation beyond
accept/reject plumbing. Keeping them in `kernel::any` made the aggregate kernel domain look broader
than the maintained CPU-first runtime.

The `x86_64` kernel surface was preserved. It has real numeric code, CMake test coverage,
benchmark hooks, and flash-attention comparison paths, so it belongs to the maintained CPU surface.
