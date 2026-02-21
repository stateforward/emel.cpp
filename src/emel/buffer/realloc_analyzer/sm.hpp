#pragma once

#include "emel/buffer/realloc_analyzer/actions.hpp"
#include "emel/buffer/realloc_analyzer/events.hpp"
#include "emel/buffer/realloc_analyzer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::buffer::realloc_analyzer {

/**
 * buffer realloc analyzer orchestration model.
 *
 * parity reference:
 * - `ggml_gallocr_node_needs_realloc(...)`
 * - `ggml_gallocr_needs_realloc(...)`
 *
 * runtime invariants:
 * - inputs are accepted only through `event::analyze`.
 * - phase progression uses anonymous transitions only (no internal queues).
 * - side effects are limited to capturing inputs and writing output pointers after dispatch.
 * - no cross-machine mutation; analysis is pure over payload + snapshot.
 *
 * guard semantics:
 * - `guard::can_analyze` validates payload shape and pointer presence.
 * - `guard::phase_ok` / `guard::phase_failed` route based on `context.phase_error`.
 *
 * action side effects:
 * - `begin_analyze` captures request inputs and clears outputs.
 * - `run_evaluate` updates `context.needs_realloc`.
 * - `begin_reset` clears error outputs; `on_reset_done` clears context.
 *
 * state purposes:
 * - `idle`: accepts `event::analyze` and `event::reset`.
 * - `validating`: validates graph/snapshot payload contracts.
 * - `evaluating`: evaluates whether realloc is required.
 * - `publishing`: finalizes phase state before output publication.
 * - `done`: success terminal before returning to `idle`.
 * - `failed`: failure terminal before returning to `idle`.
 * - `resetting`/`reset_final`: clears runtime analysis state.
 */
struct model {
 auto operator()() const {
    namespace sml = boost::sml;

    struct idle {};
    struct validating {};
    struct evaluating {};
    struct publishing {};
    struct done {};
    struct failed {};
    struct resetting {};
    struct reset_final {};

    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::analyze> [guard::can_analyze{}] /
          action::begin_analyze = sml::state<validating>,
      sml::state<idle> + sml::event<event::analyze> / action::reject_invalid =
          sml::state<failed>,

      sml::state<validating> / action::run_validate = sml::state<evaluating>,
      sml::state<evaluating> [guard::phase_failed] = sml::state<failed>,
      sml::state<evaluating> [guard::phase_ok] / action::run_evaluate =
          sml::state<publishing>,
      sml::state<publishing> [guard::phase_failed] = sml::state<failed>,
      sml::state<publishing> [guard::phase_ok] / action::run_publish = sml::state<done>,
      sml::state<done> [guard::phase_failed] = sml::state<failed>,
      sml::state<done> [guard::phase_ok] / action::on_analyze_done = sml::state<idle>,

      sml::state<failed> [guard::always] / action::on_analyze_error = sml::state<idle>,

      sml::state<idle> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<validating> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<evaluating> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<publishing> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<done> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<failed> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<resetting> / action::begin_reset = sml::state<reset_final>,
      sml::state<reset_final> [guard::phase_failed] / action::on_reset_error =
          sml::state<failed>,
      sml::state<reset_final> [guard::phase_ok] / action::on_reset_done =
          sml::state<idle>,

      sml::state<validating> + sml::event<event::analyze> / action::on_unexpected =
          sml::state<failed>,
      sml::state<evaluating> + sml::event<event::analyze> / action::on_unexpected =
          sml::state<failed>,
      sml::state<publishing> + sml::event<event::analyze> / action::on_unexpected =
          sml::state<failed>,
      sml::state<done> + sml::event<event::analyze> / action::on_unexpected =
          sml::state<failed>,
      sml::state<resetting> + sml::event<event::analyze> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reset_final> + sml::event<event::analyze> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<event::analyze> / action::on_unexpected =
          sml::state<failed>,

      sml::state<resetting> + sml::event<event::reset> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reset_final> + sml::event<event::reset> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<event::reset> / action::on_unexpected =
          sml::state<failed>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::analyze & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    if (context_.phase_error == EMEL_OK) {
      if (ev.needs_realloc_out != nullptr) {
        *ev.needs_realloc_out = context_.needs_realloc ? 1 : 0;
      }
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_OK;
      }
    } else {
      const int32_t err = action::detail::normalize_error(context_.phase_error, EMEL_ERR_BACKEND);
      if (ev.error_out != nullptr) {
        *ev.error_out = err;
      }
    }
    return emel::detail::normalize_event_result(ev, accepted);
  }

  bool needs_realloc() const noexcept { return context_.needs_realloc; }

 private:
  action::context context_{};
};

}  // namespace emel::buffer::realloc_analyzer
