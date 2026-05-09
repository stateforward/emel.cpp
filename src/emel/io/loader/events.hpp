#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/events.hpp"
#include "emel/io/loader/errors.hpp"

namespace emel::io::loader::events {

struct load_tensor_done;
struct load_tensor_error;
struct load_tensor_batch_done;
struct load_tensor_batch_error;

} // namespace emel::io::loader::events

namespace emel::io::loader::event {

inline constexpr uint64_t k_default_staged_read_chunk_bytes = 64u * 1024u;

enum class strategy_kind : uint8_t {
  none = 0u,
  mapped_file = 1u,
  read_copy = 2u,
  external_buffer = 3u,
  staged_read = 4u,
};

struct strategy_policy {
  strategy_kind strategy = strategy_kind::none;
  uint64_t staged_chunk_bytes = k_default_staged_read_chunk_bytes;
};

using tensor_load_span = emel::io::event::tensor_load_span;

struct load_tensor {
  const tensor_load_span &tensor;
  const strategy_policy &policy;
  emel::callback<void(const events::load_tensor_done &)> on_done = {};
  emel::callback<void(const events::load_tensor_error &)> on_error = {};

  load_tensor(const tensor_load_span &tensor_in,
              const strategy_policy &policy_in) noexcept
      : tensor(tensor_in), policy(policy_in) {}
};

struct load_tensor_batch {
  std::span<const tensor_load_span> tensors = {};
  const strategy_policy &policy;
  emel::callback<void(const events::load_tensor_batch_done &)> on_done = {};
  emel::callback<void(const events::load_tensor_batch_error &)> on_error = {};

  load_tensor_batch(std::span<const tensor_load_span> tensors_in,
                    const strategy_policy &policy_in) noexcept
      : tensors(tensors_in), policy(policy_in) {}
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
  emel::error::type strategy_err = emel::error::cast(error::none);
};

struct load_tensor_batch_done {
  const event::load_tensor_batch &request;
  event::strategy_kind strategy = event::strategy_kind::none;
  uint32_t done_count = 0u;
  uint64_t bytes_done = 0u;
};

struct load_tensor_batch_error {
  const event::load_tensor_batch &request;
  emel::error::type err = emel::error::cast(error::none);
  emel::error::type strategy_err = emel::error::cast(error::none);
  uint32_t failed_index = 0u;
};

} // namespace emel::io::loader::events
