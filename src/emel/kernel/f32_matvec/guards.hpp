#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "emel/kernel/f32_matvec/context.hpp"
#include "emel/kernel/f32_matvec/events.hpp"

namespace emel::kernel::f32_matvec::guard {

inline bool guard_element_count_valid(const uint64_t inner, const uint64_t rows,
                                      uint64_t &elements) noexcept {
  if (inner == 0u || rows == 0u ||
      inner > std::numeric_limits<uint64_t>::max() / rows) {
    return false;
  }
  elements = inner * rows;
  return elements <= std::numeric_limits<size_t>::max() / sizeof(float) &&
         elements <= std::numeric_limits<uintptr_t>::max() / sizeof(float);
}

inline bool guard_ranges_disjoint(const void *first, const uint64_t first_bytes,
                                  const void *second,
                                  const uint64_t second_bytes) noexcept {
  const auto first_begin = reinterpret_cast<uintptr_t>(first);
  const auto second_begin = reinterpret_cast<uintptr_t>(second);
  if (first_begin > std::numeric_limits<uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<uintptr_t>::max() - second_bytes) {
    return false;
  }
  const uintptr_t first_end = first_begin + first_bytes;
  const uintptr_t second_end = second_begin + second_bytes;
  return first_end <= second_begin || second_end <= first_begin;
}

template <class source_type>
inline bool guard_spans_disjoint(const std::span<const source_type> source,
                                 const std::span<float> destination,
                                 const uint64_t elements) noexcept {
  if (elements > std::numeric_limits<uintptr_t>::max() / sizeof(source_type) ||
      elements > std::numeric_limits<uintptr_t>::max() / sizeof(float)) {
    return false;
  }
  return guard_ranges_disjoint(source.data(), elements * sizeof(source_type),
                               destination.data(), elements * sizeof(float));
}

template <class source_type, class request_type>
inline bool guard_prepare_request_valid(const request_type &request) noexcept {
  uint64_t elements = 0u;
  if (!guard_element_count_valid(request.inner, request.rows, elements) ||
      request.source.data() == nullptr ||
      request.destination.data() == nullptr ||
      request.source.size() < elements ||
      request.destination.size() < elements) {
    return false;
  }
  return reinterpret_cast<uintptr_t>(request.source.data()) %
                 alignof(source_type) ==
             0u &&
         reinterpret_cast<uintptr_t>(request.destination.data()) %
                 alignof(float) ==
             0u &&
         guard_spans_disjoint<source_type>(request.source, request.destination,
                                           elements);
}

inline bool
guard_execute_request_valid(const event::execute_request &request) noexcept {
  uint64_t elements = 0u;
  if (!guard_element_count_valid(request.inner, request.rows, elements) ||
      request.weights.data() == nullptr || request.input.data() == nullptr ||
      request.output.data() == nullptr || request.weights.size() < elements ||
      request.input.size() < request.inner ||
      request.output.size() < request.rows ||
      reinterpret_cast<uintptr_t>(request.weights.data()) % alignof(float) !=
          0u ||
      reinterpret_cast<uintptr_t>(request.input.data()) % alignof(float) !=
          0u ||
      reinterpret_cast<uintptr_t>(request.output.data()) % alignof(float) !=
          0u) {
    return false;
  }
  const uint64_t weight_bytes = elements * sizeof(float);
  const uint64_t input_bytes = request.inner * sizeof(float);
  const uint64_t output_bytes = request.rows * sizeof(float);
  return guard_ranges_disjoint(request.weights.data(), weight_bytes,
                               request.output.data(), output_bytes) &&
         guard_ranges_disjoint(request.input.data(), input_bytes,
                               request.output.data(), output_bytes);
}

struct guard_prepare_f32_supported {
  bool operator()(const event::prepare_f32 &ev,
                  const action::context &) const noexcept {
    return guard_prepare_request_valid<float>(ev.request);
  }
};

struct guard_prepare_f32_unsupported {
  bool operator()(const event::prepare_f32 &ev,
                  const action::context &ctx) const noexcept {
    return !guard_prepare_f32_supported{}(ev, ctx);
  }
};

struct guard_prepare_f16_supported {
  bool operator()(const event::prepare_f16 &ev,
                  const action::context &) const noexcept {
    return guard_prepare_request_valid<uint16_t>(ev.request);
  }
};

struct guard_prepare_f16_unsupported {
  bool operator()(const event::prepare_f16 &ev,
                  const action::context &ctx) const noexcept {
    return !guard_prepare_f16_supported{}(ev, ctx);
  }
};

struct guard_execute_reference_supported {
  bool operator()(const event::execute_reference &ev,
                  const action::context &) const noexcept {
    return guard_execute_request_valid(ev.request);
  }
};

struct guard_execute_reference_unsupported {
  bool operator()(const event::execute_reference &ev,
                  const action::context &ctx) const noexcept {
    return !guard_execute_reference_supported{}(ev, ctx);
  }
};

struct guard_execute_exact_x4_supported {
  bool operator()(const event::execute_exact_x4 &ev,
                  const action::context &) const noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return guard_execute_request_valid(ev.request);
#else
    (void)ev;
    return false;
#endif
  }
};

struct guard_execute_exact_x4_unsupported {
  bool operator()(const event::execute_exact_x4 &ev,
                  const action::context &ctx) const noexcept {
    return !guard_execute_exact_x4_supported{}(ev, ctx);
  }
};

template <class event_type> struct guard_dispatch_succeeded {
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return ev.result.accepted;
  }
};

template <class event_type> struct guard_dispatch_failed {
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return !ev.result.accepted;
  }
};

template <class event_type> struct guard_has_done_callback {
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.on_done);
  }
};

template <class event_type> struct guard_no_done_callback {
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_done_callback<event_type>{}(ev, ctx);
  }
};

template <class event_type> struct guard_has_error_callback {
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.on_error);
  }
};

template <class event_type> struct guard_no_error_callback {
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_callback<event_type>{}(ev, ctx);
  }
};

struct guard_prepare_f32_succeeded
    : guard_dispatch_succeeded<event::prepare_f32> {};
struct guard_prepare_f32_failed : guard_dispatch_failed<event::prepare_f32> {};
struct guard_prepare_f16_succeeded
    : guard_dispatch_succeeded<event::prepare_f16> {};
struct guard_prepare_f16_failed : guard_dispatch_failed<event::prepare_f16> {};
struct guard_execute_reference_succeeded
    : guard_dispatch_succeeded<event::execute_reference> {};
struct guard_execute_reference_failed
    : guard_dispatch_failed<event::execute_reference> {};
struct guard_execute_exact_x4_succeeded
    : guard_dispatch_succeeded<event::execute_exact_x4> {};
struct guard_execute_exact_x4_failed
    : guard_dispatch_failed<event::execute_exact_x4> {};

struct guard_prepare_f32_has_done_callback
    : guard_has_done_callback<event::prepare_f32> {};
struct guard_prepare_f32_no_done_callback
    : guard_no_done_callback<event::prepare_f32> {};
struct guard_prepare_f32_has_error_callback
    : guard_has_error_callback<event::prepare_f32> {};
struct guard_prepare_f32_no_error_callback
    : guard_no_error_callback<event::prepare_f32> {};
struct guard_prepare_f16_has_done_callback
    : guard_has_done_callback<event::prepare_f16> {};
struct guard_prepare_f16_no_done_callback
    : guard_no_done_callback<event::prepare_f16> {};
struct guard_prepare_f16_has_error_callback
    : guard_has_error_callback<event::prepare_f16> {};
struct guard_prepare_f16_no_error_callback
    : guard_no_error_callback<event::prepare_f16> {};
struct guard_execute_reference_has_done_callback
    : guard_has_done_callback<event::execute_reference> {};
struct guard_execute_reference_no_done_callback
    : guard_no_done_callback<event::execute_reference> {};
struct guard_execute_reference_has_error_callback
    : guard_has_error_callback<event::execute_reference> {};
struct guard_execute_reference_no_error_callback
    : guard_no_error_callback<event::execute_reference> {};
struct guard_execute_exact_x4_has_done_callback
    : guard_has_done_callback<event::execute_exact_x4> {};
struct guard_execute_exact_x4_no_done_callback
    : guard_no_done_callback<event::execute_exact_x4> {};
struct guard_execute_exact_x4_has_error_callback
    : guard_has_error_callback<event::execute_exact_x4> {};
struct guard_execute_exact_x4_no_error_callback
    : guard_no_error_callback<event::execute_exact_x4> {};

} // namespace emel::kernel::f32_matvec::guard
