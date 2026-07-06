#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/sm.hpp"

namespace emel::text::generator::matmul {

inline constexpr size_t k_matmul_lanes = 8u;
inline constexpr size_t k_matmul_worker_lanes = k_matmul_lanes - 1u;

using lane_pool = emel::policy::fork_join_lane_pool<k_matmul_worker_lanes, 128u>;

enum class lane_mode : uint8_t {
  serial = 0,
  parallel = 1,
};

namespace detail {

struct matmul_row_slice {
  int32_t row_begin = 0;
  int32_t row_count = 0;
};

inline uint64_t matmul_slice_group_rows(const emel::kernel::event::dtype type) noexcept {
  const uint8_t code = emel::kernel::detail::dtype_code(type);
  const uint64_t x8_group =
      static_cast<uint64_t>(code == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
                            code == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
                            code == emel::kernel::detail::dtype_q6_k_x8 ||
                            code == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
                            code == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared) *
      emel::kernel::detail::quant::Q4_K_X8_ROWS;
  const uint64_t x4_group =
      static_cast<uint64_t>(code == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
                            code == emel::kernel::detail::dtype_q8_0_x4_bl8) *
      emel::kernel::detail::quant::Q8_0_X4_ROWS;
  return std::max<uint64_t>(x8_group + x4_group, 1u);
}

inline size_t compute_matmul_row_slices(
    const uint64_t rows,
    const uint64_t group_rows,
    std::array<matmul_row_slice, k_matmul_lanes> & slices) noexcept {
  const uint64_t groups = (rows + group_rows - 1u) / group_rows;
  const uint64_t lane_count = std::min<uint64_t>(k_matmul_lanes, std::max<uint64_t>(groups, 1u));
  const uint64_t groups_per_lane = groups / lane_count;
  const uint64_t extra_groups = groups % lane_count;
  uint64_t begin_group = 0u;
  for (uint64_t lane = 0u; lane < lane_count; ++lane) {
    const uint64_t lane_groups = groups_per_lane + static_cast<uint64_t>(lane < extra_groups);
    const uint64_t begin_row = begin_group * group_rows;
    const uint64_t end_row = std::min(rows, (begin_group + lane_groups) * group_rows);
    slices[lane].row_begin = static_cast<int32_t>(begin_row);
    slices[lane].row_count = static_cast<int32_t>(end_row - begin_row);
    begin_group += lane_groups;
  }
  return static_cast<size_t>(lane_count);
}

inline emel::kernel::event::op_mul_mat compute_sliced_mul_mat_event(
    const emel::kernel::event::op_mul_mat & ev,
    const uint64_t group_rows,
    const matmul_row_slice slice) noexcept {
  emel::kernel::event::op_mul_mat sliced = ev;
  const uint64_t begin = static_cast<uint64_t>(slice.row_begin);
  const uint64_t count = static_cast<uint64_t>(slice.row_count);
  const uint64_t slice_groups = (count + group_rows - 1u) / group_rows;
  sliced.src0.data =
      static_cast<const uint8_t *>(ev.src0.data) + (begin / group_rows) * ev.src0.nb[1];
  sliced.src0.ne[1] = count;
  sliced.src0.nb[2] = ev.src0.nb[1] * slice_groups;
  sliced.src0.nb[3] = sliced.src0.nb[2];
  sliced.dst.data = static_cast<uint8_t *>(ev.dst.data) + begin * ev.dst.nb[1];
  sliced.dst.ne[1] = count;
  sliced.dst.nb[2] = ev.dst.nb[1] * count;
  sliced.dst.nb[3] = sliced.dst.nb[2];
  return sliced;
}

}  // namespace detail

}  // namespace emel::text::generator::matmul
