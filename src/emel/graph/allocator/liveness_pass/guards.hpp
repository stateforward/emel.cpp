#pragma once

#include "emel/graph/allocator/liveness_pass/context.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"

namespace emel::graph::allocator::liveness_pass::guard {

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
           ev.request.graph_topology != nullptr &&
           ev.request.node_count != 0u &&
           ev.request.tensor_count != 0u &&
           ev.request.tensor_count <= ev.request.tensor_capacity;
  }
};

struct phase_invalid_request {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           (ev.request.graph_topology == nullptr ||
            ev.request.node_count == 0u ||
            ev.request.tensor_count == 0u);
  }
};

struct phase_capacity_exceeded {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           ev.request.graph_topology != nullptr &&
           ev.request.node_count != 0u &&
           ev.request.tensor_count != 0u &&
           ev.request.tensor_count > ev.request.tensor_capacity;
  }
};

}  // namespace emel::graph::allocator::liveness_pass::guard
