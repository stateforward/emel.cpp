#pragma once

#include "emel/graph/assembler/assemble_validate_pass/context.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"

namespace emel::graph::assembler::assemble_validate_pass::guard {

struct phase_done {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.request.step_plan != nullptr &&
           ev.request.output_out != nullptr &&
           ev.request.bytes_per_tensor != 0u &&
           ev.request.workspace_capacity_bytes != 0u;
  }
};

struct phase_invalid_request {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           (ev.request.step_plan == nullptr ||
            ev.request.output_out == nullptr ||
            ev.request.bytes_per_tensor == 0u ||
            ev.request.workspace_capacity_bytes == 0u);
  }
};

}  // namespace emel::graph::assembler::assemble_validate_pass::guard
