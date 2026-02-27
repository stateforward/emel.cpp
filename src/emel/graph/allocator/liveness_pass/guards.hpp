#pragma once

#include "emel/graph/allocator/liveness_pass/context.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"

namespace emel::graph::allocator::liveness_pass::guard {

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

struct phase_unclassified_failure {
  bool operator()(const allocator::event::allocate_graph_plan & ev,
                  const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(allocator::error::none) &&
           !phase_done{}(ev, ctx) &&
           !phase_invalid_request{}(ev, ctx) &&
           !phase_capacity_exceeded{}(ev, ctx);
  }
};

}  // namespace emel::graph::allocator::liveness_pass::guard
