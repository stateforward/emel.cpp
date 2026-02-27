#pragma once

#include "emel/graph/processor/alloc_step/context.hpp"
#include "emel/graph/processor/alloc_step/events.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"

namespace emel::graph::processor::alloc_step::action {

struct run_callback {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    int32_t callback_err = 0;
    const bool callback_ok = ev.request.alloc_graph(ev.request, &callback_err);
    ev.ctx.phase_callback_ok = callback_ok;
    ev.ctx.phase_callback_err = callback_err;
  }
};

struct mark_done {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::done;
    ev.ctx.err = emel::error::cast(processor::error::none);
  }
};

struct mark_failed_existing_error {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::failed;
  }
};

struct mark_failed_callback_error {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::failed;
    ev.ctx.err = static_cast<emel::error::type>(ev.ctx.phase_callback_err);
  }
};

struct mark_failed_callback_without_error {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(processor::error::kernel_failed);
  }
};

struct mark_failed_invalid_request {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.alloc_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(processor::error::invalid_request);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, const context &) const noexcept {
    if constexpr (requires { ev.ctx.alloc_outcome; ev.ctx.err; }) {
      ev.ctx.alloc_outcome = events::phase_outcome::failed;
      ev.ctx.err = emel::error::cast(processor::error::internal_error);
    }
  }
};

inline constexpr run_callback run_callback{};
inline constexpr mark_done mark_done{};
inline constexpr mark_failed_existing_error mark_failed_existing_error{};
inline constexpr mark_failed_callback_error mark_failed_callback_error{};
inline constexpr mark_failed_callback_without_error mark_failed_callback_without_error{};
inline constexpr mark_failed_invalid_request mark_failed_invalid_request{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::processor::alloc_step::action
