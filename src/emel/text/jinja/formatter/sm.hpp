#pragma once

#include <cstdint>

#include "emel/text/jinja/formatter/actions.hpp"
#include "emel/text/jinja/formatter/events.hpp"
#include "emel/text/jinja/formatter/guards.hpp"
#include "emel/sm.hpp"

namespace emel::text::jinja::formatter {

struct initialized {};
struct setup {};
struct eval_stmt {};
struct eval_expr {};
struct write_output {};
struct render_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * jinja renderer orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting render intent.
 * - `setup`: initialize context and seed statement work.
 * - `eval_stmt`: step through statements and enqueue expressions when needed.
 * - `eval_expr`: evaluate pending expressions.
 * - `write_output`: emit pending expression values.
 * - `render_decision`: branch based on phase results.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_render`/`invalid_render` validate request pointers and parameters.
 * - `phase_*` guards observe errors set by actions.
 *
 * action side effects:
 * - `begin_render`/`seed_program` prepare context for a render pass.
 * - `eval_next_stmt`/`eval_pending_expr`/`write_pending_value` execute rendering steps.
 * - `finalize_*` finalize terminal status on context.
 * - `reject_invalid_render` writes errors for invalid requests.
 * - `on_unexpected` reports sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<initialized> +
            sml::event<event::render>[guard::valid_render] / action::begin_render =
                sml::state<setup>,
        sml::state<initialized> +
            sml::event<event::render>[guard::invalid_render] /
                action::reject_invalid_render = sml::state<errored>,

        sml::state<done> + sml::event<event::render>[guard::valid_render] /
                               action::begin_render = sml::state<setup>,
        sml::state<done> + sml::event<event::render>[guard::invalid_render] /
                               action::reject_invalid_render =
            sml::state<errored>,

        sml::state<errored> + sml::event<event::render>[guard::valid_render] /
                                  action::begin_render = sml::state<setup>,
        sml::state<errored> + sml::event<event::render>[guard::invalid_render] /
                                  action::reject_invalid_render =
            sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::render>[guard::valid_render] / action::begin_render =
                sml::state<setup>,
        sml::state<unexpected> +
            sml::event<event::render>[guard::invalid_render] /
                action::reject_invalid_render = sml::state<unexpected>,

        sml::state<setup> / action::seed_program = sml::state<eval_stmt>,

        sml::state<eval_stmt>[guard::phase_failed{}] = sml::state<render_decision>,
        sml::state<eval_stmt>[guard::needs_expr{}] = sml::state<eval_expr>,
        sml::state<eval_stmt>[guard::has_stmt_work{}] / action::eval_next_stmt =
            sml::state<eval_stmt>,
        sml::state<eval_stmt>[guard::no_stmt_work{}] = sml::state<render_decision>,

        sml::state<eval_expr>[guard::has_pending_expr{}] / action::eval_pending_expr =
            sml::state<eval_expr>,
        sml::state<eval_expr>[guard::phase_failed{}] = sml::state<render_decision>,
        sml::state<eval_expr> = sml::state<write_output>,

        sml::state<write_output>[guard::phase_failed{}] = sml::state<render_decision>,
        sml::state<write_output>[guard::needs_write{}] / action::write_pending_value =
            sml::state<write_output>,
        sml::state<write_output> = sml::state<eval_stmt>,

        sml::state<render_decision>[guard::phase_ok{}] / action::finalize_done =
            sml::state<done>,
        sml::state<render_decision>[guard::phase_failed{}] / action::finalize_error =
            sml::state<errored>,

        sml::state<initialized> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<setup> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<eval_stmt> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<eval_expr> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<write_output> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<render_decision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<done> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<errored> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<unexpected> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  explicit sm(action::context & ctx) : base_type(ctx), context_(&ctx) {}

  bool process_event(const event::render & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const bool valid = guard::valid_render(ev);
    const int32_t err = ok ? EMEL_OK
                           : (context_->last_error != EMEL_OK ? context_->last_error
                                                               : EMEL_ERR_BACKEND);
    const size_t output_length = valid ? context_->output_length : 0;
    const size_t error_pos = valid ? context_->error_pos : 0;
    const bool output_truncated = valid ? (err != EMEL_OK) : false;

    if (ev.output_length != nullptr) {
      *ev.output_length = output_length;
    }
    if (ev.output_truncated != nullptr) {
      *ev.output_truncated = output_truncated;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ev.error_pos_out != nullptr) {
      *ev.error_pos_out = error_pos;
    }

    if (ok) {
      if (ev.dispatch_done) {
        ev.dispatch_done(events::rendering_done{&ev, output_length, output_truncated});
      }
    } else {
      if (ev.dispatch_error) {
        ev.dispatch_error(events::rendering_error{&ev, err, error_pos});
      }
    }

    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

 private:
  action::context * context_ = nullptr;
};

}  // namespace emel::text::jinja::formatter
