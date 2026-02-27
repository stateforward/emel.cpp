#pragma once

#include <cstdint>
#include <limits>

#include "emel/graph/allocator/liveness_pass/events.hpp"
#include "emel/graph/allocator/ordering_pass/context.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"

namespace emel::graph::allocator::ordering_pass::guard {

inline bool product_overflows_u64(const uint64_t lhs, const uint64_t rhs) noexcept {
  return lhs != 0u && rhs > (std::numeric_limits<uint64_t>::max() / lhs);
}

struct phase_done {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.liveness_outcome == liveness_pass::events::phase_outcome::done &&
           ev.ctx.required_intervals != 0u &&
           ev.ctx.required_intervals <= ev.request.interval_capacity &&
           ev.request.bytes_per_tensor != 0u &&
           !product_overflows_u64(static_cast<uint64_t>(ev.ctx.required_intervals),
                                  ev.request.bytes_per_tensor);
  }
};

struct phase_prereq_failed {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.liveness_outcome != liveness_pass::events::phase_outcome::done;
  }
};

struct phase_capacity_exceeded {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.liveness_outcome == liveness_pass::events::phase_outcome::done &&
           ev.ctx.required_intervals > ev.request.interval_capacity;
  }
};

struct phase_overflow {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.liveness_outcome == liveness_pass::events::phase_outcome::done &&
           ev.ctx.required_intervals != 0u &&
           ev.ctx.required_intervals <= ev.request.interval_capacity &&
           ev.request.bytes_per_tensor != 0u &&
           product_overflows_u64(static_cast<uint64_t>(ev.ctx.required_intervals),
                                 ev.request.bytes_per_tensor);
  }
};

struct phase_invalid_request {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.liveness_outcome == liveness_pass::events::phase_outcome::done &&
           (ev.ctx.required_intervals == 0u || ev.request.bytes_per_tensor == 0u);
  }
};

struct phase_unclassified_failure {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           !phase_done{}(ev, ctx) &&
           !phase_prereq_failed{}(ev, ctx) &&
           !phase_capacity_exceeded{}(ev, ctx) &&
           !phase_overflow{}(ev, ctx) &&
           !phase_invalid_request{}(ev, ctx);
  }
};

}  // namespace emel::graph::allocator::ordering_pass::guard
