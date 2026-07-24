#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::matmul {

using lane_pool = emel::policy::fork_join_lane_pool<7u, 128u, 1048576u>;

enum class lane_mode : uint8_t {
  serial = 0,
  parallel = 1,
};

struct execution_policy {
  lane_pool *parallel_matmul_lanes;
  emel::kernel::kernel_kind kernel_kind;
  size_t active_lanes;
  lane_mode mode = lane_mode::serial;
};

namespace detail {

// A row slice is a disjoint destination interval. Packed operands require
// row_begin to remain aligned to their physical row group.
struct matmul_row_slice {
  int32_t row_begin = 0;
  int32_t row_count = 0;
};

inline emel::kernel::event::op_mul_mat
compute_sliced_mul_mat_event(const emel::kernel::event::op_mul_mat &ev,
                             const uint64_t group_rows,
                             const matmul_row_slice slice) noexcept {
  emel::kernel::event::op_mul_mat sliced = ev;
  const uint64_t begin = static_cast<uint64_t>(slice.row_begin);
  const uint64_t count = static_cast<uint64_t>(slice.row_count);
  const uint64_t slice_groups = (count + group_rows - 1u) / group_rows;
  sliced.src0.data = static_cast<const uint8_t *>(ev.src0.data) +
                     (begin / group_rows) * ev.src0.nb[1];
  sliced.src0.ne[1] = count;
  sliced.src0.nb[2] = ev.src0.nb[1] * slice_groups;
  sliced.src0.nb[3] = sliced.src0.nb[2];
  sliced.dst.data = static_cast<uint8_t *>(ev.dst.data) + begin * ev.dst.nb[1];
  sliced.dst.ne[1] = count;
  sliced.dst.nb[2] = ev.dst.nb[1] * count;
  sliced.dst.nb[3] = sliced.dst.nb[2];
  return sliced;
}

} // namespace detail

} // namespace emel::kernel::matmul
