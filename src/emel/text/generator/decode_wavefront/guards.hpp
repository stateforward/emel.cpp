#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/text/generator/decode_wavefront/context.hpp"
#include "emel/text/generator/decode_wavefront/events.hpp"

namespace emel::text::generator::decode_wavefront::guard {

namespace detail {

inline bool compatible_key(const event::compatibility_key & lhs,
                           const event::compatibility_key & rhs) noexcept {
  return lhs.model_identity == rhs.model_identity &&
         lhs.backend_identity == rhs.backend_identity &&
         lhs.kernel_kind == rhs.kernel_kind &&
         lhs.attention == rhs.attention &&
         lhs.route == rhs.route &&
         lhs.output == rhs.output &&
         lhs.dtype_layout_contract == rhs.dtype_layout_contract &&
         lhs.quantized_contract == rhs.quantized_contract &&
         lhs.step_size == rhs.step_size &&
         lhs.token_count == rhs.token_count;
}

inline bool all_lanes_compatible(const event::run & ev) noexcept {
  const size_t lane_count = ev.lanes.size();
  if (lane_count == 0u || lane_count > event::k_max_lanes) {
    return false;
  }

  const auto & first = ev.lanes[0].key;
  for (size_t lane_index = 1u; lane_index < lane_count; ++lane_index) {
    if (!compatible_key(first, ev.lanes[lane_index].key)) {
      return false;
    }
  }
  return true;
}

inline bool valid_lane_count(const event::run & ev) noexcept {
  return ev.lanes.size() > 0u && ev.lanes.size() <= event::k_max_lanes;
}

// Parallel dispatch requires every lane-owned write surface to be disjoint.
// Opaque owners have no public extent, so their mechanically visible range is
// their identity byte. Known outputs and lifecycle buffers retain their full
// byte extent, which also catches cross-category and partial-range aliases.
struct guard_writable_range {
  uintptr_t begin = 0u;
  uintptr_t end = 0u;
};

// Native decode owns exactly three mutable lifecycle surfaces: logits plus key
// and value caches. Model weights and bound inputs are authoritative read-only
// leaf bindings. A larger mutable manifest is outside this compact admission
// contract and therefore remains on the ordered serial path.
inline constexpr size_t guard_max_mutable_lifecycle_ranges = 3u;
inline constexpr size_t guard_fixed_writable_ranges = 7u;
inline constexpr size_t guard_max_writable_ranges =
    guard_fixed_writable_ranges + guard_max_mutable_lifecycle_ranges;
inline constexpr int32_t guard_max_lifecycle_tensor_count = 65536;

struct guard_lane_writable_ranges {
  std::array<guard_writable_range, guard_max_writable_ranges> ranges{};
  size_t count = 0u;
};

inline bool guard_append_writable_range(guard_lane_writable_ranges & out,
                                        const void * const pointer,
                                        const uint64_t bytes) noexcept {
  if (pointer == nullptr || bytes == 0u ||
      out.count >= out.ranges.size()) {
    return false;
  }

  if constexpr (sizeof(uintptr_t) < sizeof(uint64_t)) {
    if (bytes > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
      return false;
    }
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
  const uintptr_t span = static_cast<uintptr_t>(bytes);
  if (span == 0u ||
      begin > std::numeric_limits<uintptr_t>::max() - span) {
    return false;
  }

  out.ranges[out.count] = guard_writable_range{begin, begin + span};
  ++out.count;
  return true;
}

inline bool guard_append_opaque_owner(guard_lane_writable_ranges & out,
                                      const void * const owner) noexcept {
  return owner == nullptr || guard_append_writable_range(out, owner, 1u);
}

inline bool guard_collect_lane_writable_ranges(
    const event::lane & lane, guard_lane_writable_ranges & out) noexcept {
  const auto & compute = lane.compute;
  if (!guard_append_writable_range(out, &lane.graph, 1u) ||
      !guard_append_writable_range(
          out, &lane.accepted,
          static_cast<uint64_t>(sizeof(lane.accepted))) ||
      !guard_append_writable_range(
          out, compute.output_out,
          static_cast<uint64_t>(sizeof(*compute.output_out))) ||
      !guard_append_opaque_owner(out, compute.compute_ctx) ||
      !guard_append_opaque_owner(out, compute.memory_sm) ||
      !guard_append_opaque_owner(out, compute.dispatch_done.object) ||
      !guard_append_opaque_owner(out, compute.dispatch_error.object) ||
      compute.lifecycle == nullptr || compute.lifecycle->tensors == nullptr ||
      compute.lifecycle->tensor_count <= 0 ||
      compute.lifecycle->tensor_count > guard_max_lifecycle_tensor_count) {
    return false;
  }

  size_t mutable_range_count = 0u;
  for (int32_t tensor_index = 0;
       tensor_index < compute.lifecycle->tensor_count; ++tensor_index) {
    const auto & binding = compute.lifecycle->tensors[tensor_index];
    if (binding.is_leaf) {
      continue;
    }
    if (mutable_range_count >= guard_max_mutable_lifecycle_ranges ||
        !guard_append_writable_range(out, binding.buffer,
                                     binding.buffer_bytes)) {
      return false;
    }
    ++mutable_range_count;
  }
  return true;
}

inline bool guard_writable_ranges_overlap(
    const guard_writable_range lhs,
    const guard_writable_range rhs) noexcept {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

inline bool guard_lane_writable_ranges_overlap(
    const guard_lane_writable_ranges & lhs,
    const guard_lane_writable_ranges & rhs) noexcept {
  for (size_t lhs_index = 0u; lhs_index < lhs.count; ++lhs_index) {
    for (size_t rhs_index = 0u; rhs_index < rhs.count; ++rhs_index) {
      if (guard_writable_ranges_overlap(lhs.ranges[lhs_index],
                                        rhs.ranges[rhs_index])) {
        return true;
      }
    }
  }
  return false;
}

// Each manifest is scanned once. Subsequent cross-lane work is capped at
// k_max_lanes^2 * guard_max_writable_ranges^2 and performs no allocation.
inline bool guard_parallel_payloads_disjoint(const event::run & ev) noexcept {
  std::array<guard_lane_writable_ranges, event::k_max_lanes> lane_ranges{};
  const size_t lane_count = ev.lanes.size();
  for (size_t lane_index = 0u; lane_index < lane_count; ++lane_index) {
    if (!guard_collect_lane_writable_ranges(ev.lanes[lane_index],
                                            lane_ranges[lane_index])) {
      return false;
    }
  }

  for (size_t lhs_index = 0u; lhs_index < lane_count; ++lhs_index) {
    for (size_t rhs_index = lhs_index + 1u; rhs_index < lane_count;
         ++rhs_index) {
      if (guard_lane_writable_ranges_overlap(lane_ranges[lhs_index],
                                             lane_ranges[rhs_index])) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace detail

struct guard_valid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::valid_lane_count(ev) && detail::all_lanes_compatible(ev);
  }
};

struct guard_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::valid_lane_count(ev);
  }
};

struct guard_single_lane {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return ev.lanes.size() == 1u;
  }
};

struct guard_multi_lane_compatible {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return ev.lanes.size() > 1u && detail::all_lanes_compatible(ev);
  }
};

struct guard_serial_dispatch {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return ctx.pool == nullptr || ev.lanes.size() == 1u ||
           !detail::guard_parallel_payloads_disjoint(ev);
  }
};

template <size_t lane_count>
struct guard_parallel_lane_count {
  static_assert(lane_count >= 2u && lane_count <= event::k_max_lanes);

  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return ctx.pool != nullptr && ev.lanes.size() == lane_count &&
           detail::guard_parallel_payloads_disjoint(ev);
  }
};

struct guard_multi_lane_incompatible {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return ev.lanes.size() > 1u && !detail::all_lanes_compatible(ev);
  }
};

template <size_t lane_index>
struct guard_lane_rejected {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !ev.lanes[lane_index].accepted;
  }
};

template <size_t lane_index>
struct guard_lane_accepted_and_last {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return ev.lanes[lane_index].accepted && ev.lanes.size() == lane_index + 1u;
  }
};

template <size_t lane_index>
struct guard_lane_accepted_and_more {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return ev.lanes[lane_index].accepted && ev.lanes.size() > lane_index + 1u;
  }
};

template <size_t lane_index>
struct guard_parallel_lane_rejected {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    if (!ev.out.all_submitted || !ev.out.joined) {
      return false;
    }
    if (ev.lanes.size() <= lane_index) {
      return false;
    }
    for (size_t index = 0u; index < lane_index; ++index) {
      if (!ev.lanes[index].accepted) {
        return false;
      }
    }
    return !ev.lanes[lane_index].accepted;
  }
};

struct guard_parallel_submission_failed {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !ev.out.all_submitted;
  }
};

struct guard_parallel_join_failed {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return ev.out.all_submitted && !ev.out.joined;
  }
};

struct guard_parallel_all_lanes_accepted {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    if (!ev.out.all_submitted || !ev.out.joined) {
      return false;
    }
    for (const auto & lane : ev.lanes) {
      if (!lane.accepted) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace emel::text::generator::decode_wavefront::guard
