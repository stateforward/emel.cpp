#pragma once

#include <array>
#include <cstddef>

#include "emel/kernel/events.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/text/generator/matmul/detail.hpp"

namespace emel::text::generator::matmul::event {

struct configure_kernel_kind {
  emel::kernel::kernel_kind kind = emel::kernel::kernel_kind::x86_64;
};

struct dispatch_result {
  size_t lane_count = 0u;
  bool all_submitted = false;
  bool joined = false;
  bool all_lanes_accepted = false;
};

struct execute_serial {
  execute_serial(const emel::kernel::event::op_mul_mat & request_ref,
                 dispatch_result & result_ref,
                 bool & accepted_ref) noexcept
    : request(request_ref), result(result_ref), accepted(accepted_ref) {}

  const emel::kernel::event::op_mul_mat & request;
  dispatch_result & result;
  bool & accepted;
};

struct execute_parallel {
  execute_parallel(const emel::kernel::event::op_mul_mat & request_ref,
                   dispatch_result & result_ref,
                   bool & accepted_ref) noexcept
    : request(request_ref), result(result_ref), accepted(accepted_ref) {}

  const emel::kernel::event::op_mul_mat & request;
  dispatch_result & result;
  bool & accepted;
};

}  // namespace emel::text::generator::matmul::event
