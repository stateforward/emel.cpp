#pragma once

#include <array>

#include "emel/kernel/sm.hpp"
#include "emel/text/generator/matmul/detail.hpp"

namespace emel::text::generator::matmul::action {

using lane_pool = emel::text::generator::matmul::lane_pool;

struct context {
  lane_pool * parallel_matmul_lanes = nullptr;
  emel::kernel::kernel_kind kernel_kind = emel::kernel::detect_host_kind();
  emel::kernel::sm kernel = {};
  std::array<emel::kernel::sm, emel::text::generator::matmul::k_matmul_lanes>
      lane_kernels = {};
};

}  // namespace emel::text::generator::matmul::action
