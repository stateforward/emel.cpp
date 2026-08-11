#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "emel/kernel/events.hpp"

// Metal kernel actor runtime.
//
// The actor owns one `metal_runtime` (created once at construction; nullptr
// when the host has no Metal device or the build cannot link Metal). Every
// dispatch is a synchronous GPU round trip: stage src0/src1 into pooled
// staging buffers, encode one compute command, commit, wait until completed,
// then copy dst back. The wait keeps the dispatch run-to-completion and
// single-writer, and the fixed pool keeps dispatches allocation-free.
//
// The MSL kernels replicate the scalar semantics of the shared kernel detail
// layer (src/emel/kernel/detail.hpp) for the op set the Mimi codec
// dispatches: op_mul_mat (f32/f16/q8_0), op_add (equal + broadcast row),
// op_unary (elu/gelu/silu/...), op_im2col (f32/f16), op_conv_transpose_1d
// (f32/f16 weights), and op_get_rows (f32/f16).
namespace emel::kernel::metal::detail {

constexpr uint64_t k_pool_slice_capacity_bytes = 4u * 1024u * 1024u;
constexpr uint32_t k_pool_slice_count = 8u;

// Per-dispatch kernel selection. Each value names one compiled MSL kernel;
// the transition rows pick the variant, never the runtime.
enum class kernel_id : uint32_t {
  mul_mat_f32 = 0,
  mul_mat_f16 = 1,
  mul_mat_q8_0 = 2,
  add = 3,
  add_broadcast_row = 4,
  unary = 5,
  im2col_f32 = 6,
  im2col_f16 = 7,
  conv_transpose_1d_f32 = 8,
  conv_transpose_1d_f16 = 9,
  get_rows_f32 = 10,
  get_rows_f16 = 11,
};

// Mirrors ::emel::kernel::event::tensor_view geometry; layout must match the
// MSL `tensor_info` struct byte for byte (both are packed 8-byte members).
// `nb` holds effective byte strides (nb[0] == 0 resolves to the dense
// element-size stride), so the MSL kernels never re-derive layout.
struct shader_tensor {
  std::array<uint64_t, 4> ne = {0, 1, 1, 1};
  std::array<uint64_t, 4> nb = {0, 0, 0, 0};
  uint32_t dtype = 0;
  uint32_t reserved = 0;
};

static_assert(sizeof(shader_tensor) == 72, "shader_tensor layout drift");

struct shader_params {
  shader_tensor src0 = {};
  shader_tensor src1 = {};
  shader_tensor dst = {};
  std::array<int32_t, 8> i32 = {};
  std::array<float, 8> f32 = {};
};

static_assert(sizeof(shader_params) == 3u * 72u + 8u * 4u + 8u * 4u,
              "shader_params layout drift");

// Byte span a tensor occupies in its staged (nb-preserving) layout.
uint64_t
tensor_storage_bytes(const ::emel::kernel::event::tensor_view &t) noexcept;
uint64_t
tensor_storage_bytes(const ::emel::kernel::event::tensor_view_mut &t) noexcept;

class metal_runtime {
public:
  metal_runtime() noexcept;
  ~metal_runtime();

  metal_runtime(const metal_runtime &) = delete;
  metal_runtime &operator=(const metal_runtime &) = delete;

  bool available() const noexcept;

  // Fixed staging pool. With run-to-completion dispatch, at most one dispatch
  // is in flight at a time, so three slices (src0/src1/dst) always suffice;
  // acquisition is O(1) free-list, never allocation.
  bool acquire_slices(uint32_t (&out)[3]) noexcept;
  void release_slices(const uint32_t (&slices)[3]) noexcept;
  void *slice_contents(uint32_t index) noexcept;

  // Encode one compute dispatch over the acquired slices and wait for it.
  // Non-failing by contract: guards validate availability, capacity, and
  // request shape before any action calls this.
  bool launch(kernel_id id, const shader_params &params,
              const uint32_t (&slices)[3], uint32_t threads) noexcept;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

//------------------------------------------------------------------------------//
// Op compute helpers (one per transition row variant; compile-time chosen by
// the actions). Each returns false only on runtime failure; guards already
// validated shape, dtypes, strides, and pool capacity.
//------------------------------------------------------------------------------//

bool run_mul_mat_f32(metal_runtime &rt,
                     const ::emel::kernel::event::op_mul_mat &request) noexcept;
bool run_mul_mat_f16(metal_runtime &rt,
                     const ::emel::kernel::event::op_mul_mat &request) noexcept;
bool run_mul_mat_q8_0(
    metal_runtime &rt,
    const ::emel::kernel::event::op_mul_mat &request) noexcept;
bool run_add(metal_runtime &rt,
             const ::emel::kernel::event::op_add &request) noexcept;
bool run_add_broadcast_row(
    metal_runtime &rt, const ::emel::kernel::event::op_add &request) noexcept;
bool run_unary(metal_runtime &rt,
               const ::emel::kernel::event::op_unary &request) noexcept;
bool run_im2col_f32(metal_runtime &rt,
                    const ::emel::kernel::event::op_im2col &request) noexcept;
bool run_im2col_f16(metal_runtime &rt,
                    const ::emel::kernel::event::op_im2col &request) noexcept;
bool run_conv_transpose_1d_f32(
    metal_runtime &rt,
    const ::emel::kernel::event::op_conv_transpose_1d &request) noexcept;
bool run_conv_transpose_1d_f16(
    metal_runtime &rt,
    const ::emel::kernel::event::op_conv_transpose_1d &request) noexcept;
bool run_get_rows_f32(
    metal_runtime &rt,
    const ::emel::kernel::event::op_get_rows &request) noexcept;
bool run_get_rows_f16(
    metal_runtime &rt,
    const ::emel::kernel::event::op_get_rows &request) noexcept;

} // namespace emel::kernel::metal::detail
