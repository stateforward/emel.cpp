#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "emel/kernel/attention/context.hpp"
#include "emel/kernel/attention/events.hpp"

namespace emel::kernel::attention::guard {

inline bool guard_ranges_disjoint(const void *first,
                                  const std::size_t first_bytes,
                                  const void *second,
                                  const std::size_t second_bytes) noexcept {
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  const auto max_address = std::numeric_limits<std::uintptr_t>::max();
  if (first_begin > max_address - first_bytes ||
      second_begin > max_address - second_bytes) {
    return false;
  }
  const auto first_end = first_begin + first_bytes;
  const auto second_end = second_begin + second_bytes;
  return first_end <= second_begin || second_end <= first_begin;
}

inline bool guard_request_shape_supported(
    const event::head_range_request &request) noexcept {
  if (request.hidden_dim <= 0 || request.head_dim <= 0 ||
      request.head_begin < 0 || request.head_end <= request.head_begin ||
      request.position_capacity <= 0 || request.physical_begin < 0 ||
      request.physical_begin >= request.position_capacity ||
      request.valid_positions <= 0 ||
      request.valid_positions > request.position_capacity) {
    return false;
  }
  const std::size_t hidden_dim = static_cast<std::size_t>(request.hidden_dim);
  const std::size_t head_dim = static_cast<std::size_t>(request.head_dim);
  const std::size_t capacity =
      static_cast<std::size_t>(request.position_capacity);
  const std::size_t head_begin = static_cast<std::size_t>(request.head_begin);
  const std::size_t head_end = static_cast<std::size_t>(request.head_end);
  const std::size_t head_count = head_end - head_begin;
  const std::size_t max_size = std::numeric_limits<std::size_t>::max();
  if (capacity > (max_size - request.layer_offset) / hidden_dim ||
      head_count > max_size / head_dim || head_end > hidden_dim / head_dim) {
    return false;
  }
  const std::size_t cache_required =
      request.layer_offset + capacity * hidden_dim;
  const std::size_t output_required = head_count * head_dim;
  if (head_dim > action::k_max_head_dim || capacity > action::k_max_context ||
      request.query.data() == nullptr || request.key_cache.data() == nullptr ||
      request.value_cache.data() == nullptr ||
      request.output.data() == nullptr || request.query.size() < hidden_dim ||
      request.key_cache.size() < cache_required ||
      request.value_cache.size() < cache_required ||
      request.output.size() < output_required ||
      reinterpret_cast<std::uintptr_t>(request.query.data()) % alignof(float) !=
          0u ||
      reinterpret_cast<std::uintptr_t>(request.key_cache.data()) %
              alignof(uint16_t) !=
          0u ||
      reinterpret_cast<std::uintptr_t>(request.value_cache.data()) %
              alignof(uint16_t) !=
          0u ||
      reinterpret_cast<std::uintptr_t>(request.output.data()) %
              alignof(float) !=
          0u ||
      hidden_dim > max_size / sizeof(float) ||
      cache_required > max_size / sizeof(uint16_t) ||
      output_required > max_size / sizeof(float)) {
    return false;
  }
  const std::size_t query_bytes = hidden_dim * sizeof(float);
  const std::size_t cache_bytes = cache_required * sizeof(uint16_t);
  const std::size_t output_bytes = output_required * sizeof(float);
  return guard_ranges_disjoint(request.output.data(), output_bytes,
                               request.query.data(), query_bytes) &&
         guard_ranges_disjoint(request.output.data(), output_bytes,
                               request.key_cache.data(), cache_bytes) &&
         guard_ranges_disjoint(request.output.data(), output_bytes,
                               request.value_cache.data(), cache_bytes);
}

struct guard_execute_supported {
  bool operator()(const event::execute &ev,
                  const action::context &) const noexcept {
    return guard_request_shape_supported(ev.request);
  }
};

struct guard_execute_unsupported {
  bool operator()(const event::execute &ev,
                  const action::context &ctx) const noexcept {
    return !guard_execute_supported{}(ev, ctx);
  }
};

struct guard_execute_succeeded {
  bool operator()(const event::execute &ev,
                  const action::context &) const noexcept {
    return ev.result.accepted;
  }
};

struct guard_execute_failed {
  bool operator()(const event::execute &ev,
                  const action::context &) const noexcept {
    return !ev.result.accepted;
  }
};

struct guard_has_done_callback {
  bool operator()(const event::execute &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.on_done);
  }
};

struct guard_no_done_callback {
  bool operator()(const event::execute &ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_done_callback{}(ev, ctx);
  }
};

struct guard_has_error_callback {
  bool operator()(const event::execute &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.on_error);
  }
};

struct guard_no_error_callback {
  bool operator()(const event::execute &ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_callback{}(ev, ctx);
  }
};

} // namespace emel::kernel::attention::guard
