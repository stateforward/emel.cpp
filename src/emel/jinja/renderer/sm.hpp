#pragma once

#include <cstdint>

#include "emel/jinja/renderer/actions.hpp"
#include "emel/jinja/renderer/events.hpp"
#include "emel/jinja/renderer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::jinja::renderer {

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
 * - `finalize_*` dispatch completion callbacks and write output metadata.
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
  using base_type::base_type;
  using base_type::process_event;
};

}  // namespace emel::jinja::renderer
