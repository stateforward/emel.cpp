#pragma once

#include <array>
#include <cstddef>

#include "emel/kernel/sm.hpp"
#include "emel/text/generator/matmul/detail.hpp"

namespace emel::text::generator::matmul::action {

using lane_pool_ref = emel::text::generator::matmul::lane_pool_ref;

struct context {
  lane_pool_ref parallel_matmul_lanes = {};
  emel::kernel::kernel_kind kernel_kind = emel::kernel::kernel_kind::x86_64;
  size_t active_lanes = 1u;
  size_t lane_capacity = 0u;
  emel::kernel::sm kernel = {};
  std::array<emel::kernel::sm, emel::text::generator::matmul::k_max_matmul_lanes>
      lane_kernels = {};
  std::array<emel::text::generator::matmul::detail::matmul_row_slice,
             emel::text::generator::matmul::k_max_matmul_lanes>
      row_slices = {};
  std::array<emel::kernel::event::op_mul_mat,
             emel::text::generator::matmul::k_max_matmul_lanes>
      lane_events = {};
  std::array<emel::text::generator::matmul::detail::lane_dispatch,
             emel::text::generator::matmul::k_max_matmul_lanes>
      lane_dispatches = {};
};

}  // namespace emel::text::generator::matmul::action
