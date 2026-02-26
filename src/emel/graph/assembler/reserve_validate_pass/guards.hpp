#pragma once

#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/reserve_validate_pass/context.hpp"

namespace emel::graph::assembler::reserve_validate_pass::guard {

struct phase_done {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.request.model_topology != nullptr &&
           ev.request.output_out != nullptr &&
           ev.request.max_node_count != 0u &&
           ev.request.max_tensor_count != 0u &&
           ev.request.bytes_per_tensor != 0u &&
           ev.request.workspace_capacity_bytes != 0u;
  }
};

struct phase_invalid_request {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           (ev.request.model_topology == nullptr ||
            ev.request.output_out == nullptr ||
            ev.request.max_node_count == 0u ||
            ev.request.max_tensor_count == 0u ||
            ev.request.bytes_per_tensor == 0u ||
            ev.request.workspace_capacity_bytes == 0u);
  }
};

}  // namespace emel::graph::assembler::reserve_validate_pass::guard
