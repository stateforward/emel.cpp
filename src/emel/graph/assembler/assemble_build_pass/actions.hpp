#pragma once

#include "emel/graph/assembler/assemble_build_pass/context.hpp"
#include "emel/graph/assembler/assemble_build_pass/events.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"

namespace emel::graph::assembler::assemble_build_pass::action {

struct mark_done {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.build_outcome = events::phase_outcome::done;
    ev.ctx.err = emel::error::cast(assembler::error::none);
  }
};

struct mark_failed_prereq {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.build_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::internal_error);
  }
};

struct mark_failed_capacity {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.build_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::capacity);
  }
};

struct mark_failed_invalid_request {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.build_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::invalid_request);
  }
};

struct on_unexpected {
  void operator()(const assembler::event::assemble_graph & ev, const context &) const noexcept {
    ev.ctx.build_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::internal_error);
  }

  template <class event_type>
  void operator()(const event_type &, const context &) const noexcept {
  }
};

inline constexpr mark_done mark_done{};
inline constexpr mark_failed_prereq mark_failed_prereq{};
inline constexpr mark_failed_capacity mark_failed_capacity{};
inline constexpr mark_failed_invalid_request mark_failed_invalid_request{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::assembler::assemble_build_pass::action
