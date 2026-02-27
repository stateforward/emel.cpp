#pragma once

#include <cstdint>

#include "emel/graph/allocator/context.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"

namespace emel::graph::allocator::action {

inline void reset_plan(event::allocation_plan & plan) noexcept {
  plan.tensor_count = 0;
  plan.interval_count = 0;
  plan.required_buffer_bytes = 0;
}

struct reject_invalid_allocate_with_dispatch {
  void operator()(const event::allocate_graph_plan & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_plan(*ev.request.plan_out);
    ev.request.dispatch_error(events::allocation_error{
      *ev.request.plan_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_allocate_with_output_only {
  void operator()(const event::allocate_graph_plan & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_plan(*ev.request.plan_out);
  }
};

struct reject_invalid_allocate_without_output {
  void operator()(const event::allocate_graph_plan & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct begin_allocate {
  void operator()(const event::allocate_graph_plan & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.liveness_outcome = liveness_pass::events::phase_outcome::unknown;
    ev.ctx.ordering_outcome = ordering_pass::events::phase_outcome::unknown;
    ev.ctx.placement_outcome = placement_pass::events::phase_outcome::unknown;
    ev.ctx.required_intervals = 0;
    ev.ctx.sorted_tensor_count = 0;
    ev.ctx.required_buffer_bytes = 0;
    ++ctx.dispatch_generation;
    reset_plan(*ev.request.plan_out);
  }
};

struct commit_plan {
  void operator()(const event::allocate_graph_plan & ev, const context &) const noexcept {
    ev.request.plan_out->tensor_count = ev.ctx.sorted_tensor_count;
    ev.request.plan_out->interval_count = ev.ctx.required_intervals;
    ev.request.plan_out->required_buffer_bytes = ev.ctx.required_buffer_bytes;
  }
};

struct dispatch_done {
  void operator()(const event::allocate_graph_plan & ev, const context &) const noexcept {
    ev.request.dispatch_done(events::allocation_done{*ev.request.plan_out});
  }
};

struct dispatch_error {
  void operator()(const event::allocate_graph_plan & ev, const context &) const noexcept {
    ev.request.dispatch_error(events::allocation_error{
      *ev.request.plan_out,
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

inline constexpr reject_invalid_allocate_with_dispatch reject_invalid_allocate_with_dispatch{};
inline constexpr reject_invalid_allocate_with_output_only reject_invalid_allocate_with_output_only{};
inline constexpr reject_invalid_allocate_without_output reject_invalid_allocate_without_output{};
inline constexpr begin_allocate begin_allocate{};
inline constexpr commit_plan commit_plan{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::allocator::action
