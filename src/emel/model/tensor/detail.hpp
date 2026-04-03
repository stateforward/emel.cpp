#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/model/tensor/errors.hpp"
#include "emel/model/tensor/events.hpp"

namespace emel::model::tensor::detail {

inline constexpr int32_t max_tensors = emel::model::data::k_max_tensors;

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

template <class value_type>
value_type & bind_or_sink(value_type * ptr, value_type & sink) noexcept {
  value_type * choices[2] = {&sink, ptr};
  return *choices[static_cast<size_t>(ptr != nullptr)];
}

struct runtime_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  bool accepted = false;
};

struct bind_tensor_runtime {
  const event::bind_tensor & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

struct evict_tensor_runtime {
  const event::evict_tensor & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

struct capture_tensor_state_runtime {
  const event::capture_tensor_state & request;
  runtime_status & ctx;
  int32_t & error_code_out;
};

struct tensor_storage {
  std::array<event::lifecycle, static_cast<size_t>(max_tensors)> lifecycle = {};
  std::array<const void *, static_cast<size_t>(max_tensors)> buffer = {};
  std::array<uint64_t, static_cast<size_t>(max_tensors)> buffer_bytes = {};
  std::array<uint64_t, static_cast<size_t>(max_tensors)> file_offset = {};
  std::array<uint64_t, static_cast<size_t>(max_tensors)> data_size = {};
  std::array<uint16_t, static_cast<size_t>(max_tensors)> file_index = {};
  std::array<int32_t, static_cast<size_t>(max_tensors)> tensor_type = {};
};

}  // namespace emel::model::tensor::detail
