#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/callback.hpp"

namespace emel::kernel::attention::events {

struct execute_done;
struct execute_error;

} // namespace emel::kernel::attention::events

namespace emel::kernel::attention::event {

struct head_range_request {
  std::span<const float> query = {};
  std::span<const uint16_t> key_cache = {};
  std::span<const uint16_t> value_cache = {};
  std::span<float> output = {};
  std::size_t layer_offset = 0u;
  int32_t hidden_dim = 0;
  int32_t head_dim = 0;
  int32_t head_begin = 0;
  int32_t head_end = 0;
  int32_t position_capacity = 0;
  int32_t physical_begin = 0;
  int32_t valid_positions = 0;
};

struct dispatch_result {
  bool accepted = false;
};

struct execute {
  execute(const head_range_request &request_ref,
          dispatch_result &result_ref) noexcept
      : request(request_ref), result(result_ref) {}

  const head_range_request &request;
  dispatch_result &result;
  emel::callback<void(const events::execute_done &)> on_done = {};
  emel::callback<void(const events::execute_error &)> on_error = {};
};

} // namespace emel::kernel::attention::event

namespace emel::kernel::attention::events {

struct execute_done {
  const event::execute *request = nullptr;
};

struct execute_error {
  const event::execute *request = nullptr;
};

} // namespace emel::kernel::attention::events
