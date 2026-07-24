#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/callback.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/matmul/detail.hpp"
#include "emel/kernel/sm.hpp"

namespace emel::kernel::matmul::events {

template <class request_type> struct execute_done {
  const request_type &request;
};

template <class request_type> struct execute_error {
  const request_type &request;
};

} // namespace emel::kernel::matmul::events

namespace emel::kernel::matmul::event {

struct configure_kernel_kind {
  emel::kernel::kernel_kind kind = emel::kernel::kernel_kind::x86_64;
};

struct dispatch_result {
  size_t lane_count = 0u;
  size_t submitted_worker_lanes = 0u;
  size_t drained_worker_lanes = 0u;
  bool all_submitted = false;
  bool all_lanes_accepted = false;
};

struct diagnostics : emel::kernel::event::diagnostics {
  uint64_t serial_optimized_q4_dispatch_calls = 0u;
  uint64_t parallel_optimized_q4_dispatch_calls = 0u;
};

struct capture_diagnostics {
  capture_diagnostics(diagnostics &out_ref, bool &accepted_ref) noexcept
      : out(out_ref), accepted(accepted_ref) {}

  diagnostics &out;
  bool &accepted;
};

struct execute_serial {
  execute_serial(const emel::kernel::event::op_mul_mat &request_ref,
                 dispatch_result &result_ref, bool &accepted_ref) noexcept
      : request(request_ref), result(result_ref), accepted(accepted_ref) {}

  const emel::kernel::event::op_mul_mat &request;
  dispatch_result &result;
  bool &accepted;
  emel::callback<void(const events::execute_done<execute_serial> &)> on_done =
      {};
  emel::callback<void(const events::execute_error<execute_serial> &)> on_error =
      {};
};

struct execute_parallel {
  execute_parallel(const emel::kernel::event::op_mul_mat &request_ref,
                   dispatch_result &result_ref, bool &accepted_ref) noexcept
      : request(request_ref), result(result_ref), accepted(accepted_ref) {}

  const emel::kernel::event::op_mul_mat &request;
  dispatch_result &result;
  bool &accepted;
  emel::callback<void(const events::execute_done<execute_parallel> &)> on_done =
      {};
  emel::callback<void(const events::execute_error<execute_parallel> &)>
      on_error = {};
};

} // namespace emel::kernel::matmul::event
