#pragma once

#include <cstdint>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/loader/errors.hpp"

namespace emel::io::loader::events {

struct load_tensor_done;
struct load_tensor_error;

} // namespace emel::io::loader::events

namespace emel::io::loader::event {

enum class strategy_kind : uint8_t {
  none = 0u,
  mapped_file = 1u,
  staged_read = 2u,
  external_buffer = 3u,
};

struct strategy_policy {
  strategy_kind strategy = strategy_kind::none;
};

struct tensor_load_span {
  int32_t tensor_id = 0;
  uint16_t file_index = 0u;
  uint64_t file_offset = 0u;
  uint64_t byte_size = 0u;
  void *target = nullptr;
};

struct load_tensor {
  const tensor_load_span &tensor;
  const strategy_policy &policy;
  emel::callback<void(const events::load_tensor_done &)> on_done = {};
  emel::callback<void(const events::load_tensor_error &)> on_error = {};

  load_tensor(const tensor_load_span &tensor_in,
              const strategy_policy &policy_in) noexcept
      : tensor(tensor_in), policy(policy_in) {}
};

} // namespace emel::io::loader::event

namespace emel::io::loader::events {

struct load_tensor_done {
  const event::load_tensor &request;
  event::strategy_kind strategy = event::strategy_kind::none;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
};

struct load_tensor_error {
  const event::load_tensor &request;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::io::loader::events
