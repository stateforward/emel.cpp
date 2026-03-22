#pragma once

#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"
#include "emel/graph/processor/detail.hpp"
#include "emel/graph/processor/extract_step/context.hpp"
#include "emel/graph/processor/extract_step/events.hpp"

namespace emel::graph::processor::extract_step::action {

struct run_callback {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    int32_t outputs_produced = 0;
    int32_t callback_err = 0;
    const bool callback_ok =
        ev.request.extract_outputs(ev.request, &outputs_produced, &callback_err);
    ev.ctx.outputs_produced = outputs_produced;
    ev.ctx.phase_callback_ok = callback_ok;
    ev.ctx.phase_callback_err = callback_err;
  }
};

struct mark_done {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.extract_outcome = events::phase_outcome::done;
    ev.ctx.err = emel::error::cast(processor::error::none);
  }
};

struct mark_failed_existing_error {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.extract_outcome = events::phase_outcome::failed;
    (void)::emel::graph::processor::detail::release_phase_tensors(ev.request);
  }
};

struct mark_failed_callback_error {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.extract_outcome = events::phase_outcome::failed;
    ev.ctx.err = static_cast<emel::error::type>(ev.ctx.phase_callback_err);
    (void)::emel::graph::processor::detail::release_phase_tensors(ev.request);
  }
};

struct mark_failed_callback_without_error {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.extract_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(processor::error::kernel_failed);
    (void)::emel::graph::processor::detail::release_phase_tensors(ev.request);
  }
};

struct mark_failed_invalid_request {
  void operator()(const processor::event::execute_step & ev, context &) const noexcept {
    ev.ctx.extract_outcome = events::phase_outcome::failed;
    ev.ctx.err = emel::error::cast(processor::error::invalid_request);
    (void)::emel::graph::processor::detail::release_phase_tensors(ev.request);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, const context &) const noexcept {
    if constexpr (requires { ev.ctx.extract_outcome; ev.ctx.err; }) {
      ev.ctx.extract_outcome = events::phase_outcome::failed;
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

}  // namespace emel::graph::processor::extract_step::action
