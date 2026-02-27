#pragma once

#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/reuse_decision_pass/context.hpp"
#include "emel/graph/assembler/reuse_decision_pass/events.hpp"

namespace emel::graph::assembler::reuse_decision_pass::action {

struct mark_reuse {
  void operator()(const assembler::event::assemble_graph & ev, context & ctx) const noexcept {
    ev.ctx.reuse_outcome = events::phase_outcome::reused;
    ev.ctx.reused_topology = 1u;
    ev.ctx.assembled_node_count = ctx.reserved_node_count;
    ev.ctx.assembled_tensor_count = ctx.reserved_tensor_count;
    ev.ctx.alloc_plan = allocator::event::allocation_plan{
      .tensor_count = ctx.reserved_tensor_count,
      .interval_count = ctx.reserved_tensor_count,
      .required_buffer_bytes = ctx.reserved_required_buffer_bytes,
    };
    ev.ctx.err = emel::error::cast(assembler::error::none);
  }
};

struct mark_rebuild {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.reuse_outcome = events::phase_outcome::rebuild;
    ev.ctx.reused_topology = 0u;
    ev.ctx.assembled_node_count = ev.request.node_count_hint;
    ev.ctx.assembled_tensor_count = ev.request.tensor_count_hint;
    ev.ctx.alloc_plan = {};
    ev.ctx.err = emel::error::cast(assembler::error::none);
  }
};

struct mark_failed_prereq {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.reuse_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::internal_error);
  }
};

struct mark_failed_invalid_request {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.reuse_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::invalid_request);
  }
};

struct on_unexpected {
  void operator()(const assembler::event::assemble_graph & ev, const context &) const noexcept {
    ev.ctx.reuse_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::internal_error);
  }

  template <class event_type>
  void operator()(const event_type &, const context &) const noexcept {
  }
};

inline constexpr mark_reuse mark_reuse{};
inline constexpr mark_rebuild mark_rebuild{};
inline constexpr mark_failed_prereq mark_failed_prereq{};
inline constexpr mark_failed_invalid_request mark_failed_invalid_request{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::assembler::reuse_decision_pass::action
