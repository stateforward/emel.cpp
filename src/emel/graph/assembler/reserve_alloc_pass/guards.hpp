#pragma once

#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/reserve_alloc_pass/context.hpp"
#include "emel/graph/assembler/reserve_build_pass/events.hpp"

namespace emel::graph::assembler::reserve_alloc_pass::guard {

struct phase_request_allocator {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.build_outcome == reserve_build_pass::events::phase_outcome::done &&
           ev.request.model_topology != nullptr &&
           ev.request.output_out != nullptr &&
           ev.ctx.assembled_node_count != 0u &&
           ev.ctx.assembled_tensor_count != 0u;
  }
};

struct phase_prereq_failed {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.build_outcome != reserve_build_pass::events::phase_outcome::done;
  }
};

struct phase_invalid_request {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.build_outcome == reserve_build_pass::events::phase_outcome::done &&
           (ev.request.model_topology == nullptr ||
            ev.request.output_out == nullptr ||
            ev.ctx.assembled_node_count == 0u ||
            ev.ctx.assembled_tensor_count == 0u);
  }
};

}  // namespace emel::graph::assembler::reserve_alloc_pass::guard
