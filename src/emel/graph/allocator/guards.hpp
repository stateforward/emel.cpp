#pragma once

#include "emel/graph/allocator/context.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"

namespace emel::graph::allocator::guard {

struct valid_allocate {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.request.graph_topology != nullptr &&
           ev.request.plan_out != nullptr &&
           ev.request.node_count != 0u &&
           ev.request.tensor_count != 0u &&
           ev.request.tensor_capacity != 0u &&
           ev.request.interval_capacity != 0u &&
           ev.request.bytes_per_tensor != 0u &&
           ev.request.workspace_capacity_bytes != 0u &&
           static_cast<bool>(ev.request.dispatch_done) &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_allocate {
  bool operator()(const event::allocate_graph_plan & ev, const action::context & ctx) const noexcept {
    return !valid_allocate{}(ev, ctx);
  }
};

struct invalid_allocate_with_dispatchable_output {
  bool operator()(const event::allocate_graph_plan & ev, const action::context & ctx) const noexcept {
    return invalid_allocate{}(ev, ctx) &&
           ev.request.plan_out != nullptr &&
           static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_allocate_with_output_only {
  bool operator()(const event::allocate_graph_plan & ev, const action::context & ctx) const noexcept {
    return invalid_allocate{}(ev, ctx) &&
           ev.request.plan_out != nullptr &&
           !static_cast<bool>(ev.request.dispatch_error);
  }
};

struct invalid_allocate_without_output {
  bool operator()(const event::allocate_graph_plan & ev, const action::context & ctx) const noexcept {
    return invalid_allocate{}(ev, ctx) && ev.request.plan_out == nullptr;
  }
};

struct phase_ok {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct phase_failed {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct liveness_done {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.liveness_outcome == liveness_pass::events::phase_outcome::done;
  }
};

struct liveness_failed {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.liveness_outcome == liveness_pass::events::phase_outcome::failed;
  }
};

struct ordering_done {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.ordering_outcome == ordering_pass::events::phase_outcome::done;
  }
};

struct ordering_failed {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.ordering_outcome == ordering_pass::events::phase_outcome::failed;
  }
};

struct placement_done {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none) &&
           ev.ctx.placement_outcome == placement_pass::events::phase_outcome::done;
  }
};

struct placement_failed {
  bool operator()(const event::allocate_graph_plan & ev, const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none) ||
           ev.ctx.placement_outcome == placement_pass::events::phase_outcome::failed;
  }
};

}  // namespace emel::graph::allocator::guard
