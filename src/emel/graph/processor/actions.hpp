#pragma once

#include <cstdint>

#include "emel/graph/processor/context.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"

namespace emel::graph::processor::action {

inline void reset_output(event::execution_output & output) noexcept {
  output.outputs_produced = 0;
  output.graph_reused = 0;
  output.lifecycle = nullptr;
}

struct reject_invalid_execute_with_dispatch {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_output(*ev.request.output_out);
    ev.request.dispatch_error(events::execution_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_execute_with_output_only {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_output(*ev.request.output_out);
  }
};

struct reject_invalid_execute_without_output {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct begin_execute {
  void operator()(const event::execute_step & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.validate_outcome = validate_step::events::phase_outcome::unknown;
    ev.ctx.prepare_outcome = prepare_step::events::phase_outcome::unknown;
    ev.ctx.alloc_outcome = alloc_step::events::phase_outcome::unknown;
    ev.ctx.bind_outcome = bind_step::events::phase_outcome::unknown;
    ev.ctx.kernel_outcome = kernel_step::events::phase_outcome::unknown;
    ev.ctx.extract_outcome = extract_step::events::phase_outcome::unknown;
    ev.ctx.graph_reused = 0;
    ev.ctx.outputs_produced = 0;
    ev.ctx.phase_callback_ok = false;
    ev.ctx.phase_callback_err = 0;
    ++ctx.dispatch_generation;
    reset_output(*ev.request.output_out);
  }
};

struct commit_output {
  void operator()(const event::execute_step & ev, const context &) const noexcept {
    ev.request.output_out->outputs_produced = ev.ctx.outputs_produced;
    ev.request.output_out->graph_reused = ev.ctx.graph_reused;
    ev.request.output_out->lifecycle = ev.request.lifecycle;
  }
};

struct dispatch_done {
  void operator()(const event::execute_step & ev, const context &) const noexcept {
    ev.request.dispatch_done(events::execution_done{*ev.request.output_out});
  }
};

struct dispatch_error {
  void operator()(const event::execute_step & ev, const context &) const noexcept {
    ev.request.dispatch_error(events::execution_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr reject_invalid_execute_with_dispatch reject_invalid_execute_with_dispatch{};
inline constexpr reject_invalid_execute_with_output_only reject_invalid_execute_with_output_only{};
inline constexpr reject_invalid_execute_without_output reject_invalid_execute_without_output{};
inline constexpr begin_execute begin_execute{};
inline constexpr commit_output commit_output{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::processor::action
