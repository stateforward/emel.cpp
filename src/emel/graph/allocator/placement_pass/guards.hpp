#pragma once

#include "emel/graph/allocator/ordering_pass/events.hpp"
#include "emel/graph/allocator/placement_pass/context.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"

namespace emel::graph::allocator::placement_pass::guard {

struct phase_prefailed {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(allocator::error::none);
  }
};

struct phase_done {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.ordering_outcome == ordering_pass::events::phase_outcome::done &&
           ev.request.plan_out != nullptr &&
           ev.ctx.sorted_tensor_count != 0u &&
           ev.ctx.required_buffer_bytes <= ev.request.workspace_capacity_bytes;
  }
};

struct phase_prereq_failed {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.ordering_outcome != ordering_pass::events::phase_outcome::done;
  }
};

struct phase_capacity_exceeded {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.ordering_outcome == ordering_pass::events::phase_outcome::done &&
           ev.ctx.required_buffer_bytes > ev.request.workspace_capacity_bytes;
  }
};

struct phase_invalid_request {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.ctx.ordering_outcome == ordering_pass::events::phase_outcome::done &&
           (ev.request.plan_out == nullptr || ev.ctx.sorted_tensor_count == 0u);
  }
};

}  // namespace emel::graph::allocator::placement_pass::guard
