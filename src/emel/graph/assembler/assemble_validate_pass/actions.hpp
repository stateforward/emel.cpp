#pragma once

#include "emel/graph/assembler/assemble_validate_pass/context.hpp"
#include "emel/graph/assembler/assemble_validate_pass/events.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"

namespace emel::graph::assembler::assemble_validate_pass::action {

struct mark_done {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.validate_outcome = events::phase_outcome::done;
    ev.ctx.err = emel::error::cast(assembler::error::none);
  }
};

struct mark_failed_invalid_request {
  void operator()(const assembler::event::assemble_graph & ev, context &) const noexcept {
    ev.ctx.validate_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::invalid_request);
  }
};

struct on_unexpected {
  void operator()(const assembler::event::assemble_graph & ev, const context &) const noexcept {
    ev.ctx.validate_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(assembler::error::internal_error);
  }

  template <class event_type>
  void operator()(const event_type &, const context &) const noexcept {
  }
};

inline constexpr mark_done mark_done{};
inline constexpr mark_failed_invalid_request mark_failed_invalid_request{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::assembler::assemble_validate_pass::action
