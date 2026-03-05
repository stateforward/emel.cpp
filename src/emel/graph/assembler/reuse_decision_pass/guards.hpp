#pragma once

#include "emel/graph/assembler/assemble_validate_pass/events.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/reuse_decision_pass/context.hpp"

namespace emel::graph::assembler::reuse_decision_pass::guard {

struct phase_prefailed {
  bool operator()(const assembler::event::assemble_graph & ev,
                  const action::context &) const noexcept {
    return ev.ctx.err != emel::error::cast(assembler::error::none);
  }
};

struct phase_reuse {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context & ctx) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.validate_outcome == assemble_validate_pass::events::phase_outcome::done &&
           ctx.has_reserved_topology != 0u &&
           ctx.reserved_topology != nullptr &&
           ev.request.node_count_hint == ctx.reserved_node_count &&
           ev.request.tensor_count_hint == ctx.reserved_tensor_count;
  }
};

struct phase_rebuild {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context & ctx) const noexcept {
    const bool reuse_candidate = ctx.has_reserved_topology != 0u &&
                                 ctx.reserved_topology != nullptr &&
                                 ev.request.node_count_hint == ctx.reserved_node_count &&
                                 ev.request.tensor_count_hint == ctx.reserved_tensor_count;
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.validate_outcome == assemble_validate_pass::events::phase_outcome::done &&
           !reuse_candidate &&
           ev.request.node_count_hint != 0u &&
           ev.request.tensor_count_hint != 0u;
  }
};

struct phase_prereq_failed {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.validate_outcome != assemble_validate_pass::events::phase_outcome::done;
  }
};

struct phase_invalid_request {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context & ctx) const noexcept {
    const bool reuse_candidate = ctx.has_reserved_topology != 0u &&
                                 ctx.reserved_topology != nullptr &&
                                 ev.request.node_count_hint == ctx.reserved_node_count &&
                                 ev.request.tensor_count_hint == ctx.reserved_tensor_count;
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.validate_outcome == assemble_validate_pass::events::phase_outcome::done &&
           !reuse_candidate &&
           (ev.request.node_count_hint == 0u || ev.request.tensor_count_hint == 0u);
  }
};

}  // namespace emel::graph::assembler::reuse_decision_pass::guard
