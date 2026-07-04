#pragma once

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

// Parallel dispatch requires one graph actor per lane: concurrent
// process_event on a shared actor would break the RTC single-writer
// contract. Lane count is bounded by k_max_lanes, so the pairwise scan is
// statically bounded.
inline bool all_lane_graphs_distinct(const event::run & ev) noexcept {
  const size_t lane_count = ev.lanes.size();
  for (size_t i = 0u; i < lane_count; ++i) {
    for (size_t j = i + 1u; j < lane_count; ++j) {
      if (&ev.lanes[i].graph == &ev.lanes[j].graph) {
        return false;
      }
    }
  }
  return true;
}

// Parallel dispatch also requires one caller-owned outcome slot per lane:
// concurrent workers writing a shared `accepted` bool would race and make
// the post-join guards misreport (or hide) the failed lane. Aliased slots
// stay on the serial path, where each lane's outcome is routed before the
// next lane writes.
inline bool all_lane_outcomes_distinct(const event::run & ev) noexcept {
  const size_t lane_count = ev.lanes.size();
  for (size_t i = 0u; i < lane_count; ++i) {
    for (size_t j = i + 1u; j < lane_count; ++j) {
      if (&ev.lanes[i].accepted == &ev.lanes[j].accepted) {
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
           !detail::all_lane_graphs_distinct(ev) ||
           !detail::all_lane_outcomes_distinct(ev);
  }
};

struct guard_parallel_dispatch {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return ctx.pool != nullptr && ev.lanes.size() > 1u &&
           detail::all_lane_graphs_distinct(ev) &&
           detail::all_lane_outcomes_distinct(ev);
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

struct guard_parallel_all_lanes_accepted {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    for (const auto & lane : ev.lanes) {
      if (!lane.accepted) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace emel::text::generator::decode_wavefront::guard
