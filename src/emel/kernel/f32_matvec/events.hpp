#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"

namespace emel::kernel::f32_matvec::events {

template <class request_type> struct dispatch_done {
  const request_type &request;
};

template <class request_type> struct dispatch_error {
  const request_type &request;
};

} // namespace emel::kernel::f32_matvec::events

namespace emel::kernel::f32_matvec::event {

struct dispatch_result {
  bool accepted = false;
};

struct diagnostics {
  uint64_t prepare_calls = 0u;
  uint64_t prepared_floats = 0u;
  uint64_t reference_calls = 0u;
  uint64_t exact_x4_calls = 0u;
};

struct prepare_f32_request {
  std::span<const float> source = {};
  std::span<float> destination = {};
  uint64_t inner = 0u;
  uint64_t rows = 0u;
};

struct prepare_f16_request {
  std::span<const uint16_t> source = {};
  std::span<float> destination = {};
  uint64_t inner = 0u;
  uint64_t rows = 0u;
};

struct execute_request {
  std::span<const float> weights = {};
  std::span<const float> input = {};
  std::span<float> output = {};
  uint64_t inner = 0u;
  uint64_t rows = 0u;
};

struct prepare_f32 {
  prepare_f32(const prepare_f32_request &request_ref,
              dispatch_result &result_ref) noexcept
      : request(request_ref), result(result_ref) {}

  const prepare_f32_request &request;
  dispatch_result &result;
  emel::callback<void(const events::dispatch_done<prepare_f32> &)> on_done = {};
  emel::callback<void(const events::dispatch_error<prepare_f32> &)> on_error =
      {};
};

struct prepare_f16 {
  prepare_f16(const prepare_f16_request &request_ref,
              dispatch_result &result_ref) noexcept
      : request(request_ref), result(result_ref) {}

  const prepare_f16_request &request;
  dispatch_result &result;
  emel::callback<void(const events::dispatch_done<prepare_f16> &)> on_done = {};
  emel::callback<void(const events::dispatch_error<prepare_f16> &)> on_error =
      {};
};

struct execute_reference {
  execute_reference(const execute_request &request_ref,
                    dispatch_result &result_ref) noexcept
      : request(request_ref), result(result_ref) {}

  const execute_request &request;
  dispatch_result &result;
  emel::callback<void(const events::dispatch_done<execute_reference> &)>
      on_done = {};
  emel::callback<void(const events::dispatch_error<execute_reference> &)>
      on_error = {};
};

struct execute_exact_x4 {
  execute_exact_x4(const execute_request &request_ref,
                   dispatch_result &result_ref) noexcept
      : request(request_ref), result(result_ref) {}

  const execute_request &request;
  dispatch_result &result;
  emel::callback<void(const events::dispatch_done<execute_exact_x4> &)>
      on_done = {};
  emel::callback<void(const events::dispatch_error<execute_exact_x4> &)>
      on_error = {};
};

struct capture_diagnostics {
  capture_diagnostics(diagnostics &out_ref,
                      dispatch_result &result_ref) noexcept
      : out(out_ref), result(result_ref) {}

  diagnostics &out;
  dispatch_result &result;
};

} // namespace emel::kernel::f32_matvec::event
