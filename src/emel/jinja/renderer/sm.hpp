#pragma once

#include <cstdint>

#include "emel/jinja/renderer/actions.hpp"
#include "emel/jinja/renderer/events.hpp"
#include "emel/jinja/renderer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::jinja::renderer {

struct Initialized {};
struct Setup {};
struct EvalStmt {};
struct EvalExpr {};
struct WriteOutput {};
struct RenderDecision {};
struct Done {};
struct Errored {};
struct Unexpected {};

/**
 * Jinja renderer orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting render intent.
 * - `setup`: initialize context and seed statement work.
 * - `eval_stmt`: step through statements and enqueue expressions when needed.
 * - `eval_expr`: evaluate pending expressions.
 * - `write_output`: emit pending expression values.
 * - `render_decision`: branch based on phase results.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * Guard semantics:
 * - `valid_render`/`invalid_render` validate request pointers and parameters.
 * - `phase_*` guards observe errors set by actions.
 *
 * Action side effects:
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
        *sml::state<Initialized> +
            sml::event<event::render>[guard::valid_render] / action::begin_render =
                sml::state<Setup>,
        sml::state<Initialized> +
            sml::event<event::render>[guard::invalid_render] /
                action::reject_invalid_render = sml::state<Errored>,

        sml::state<Done> + sml::event<event::render>[guard::valid_render] /
                               action::begin_render = sml::state<Setup>,
        sml::state<Done> + sml::event<event::render>[guard::invalid_render] /
                               action::reject_invalid_render =
            sml::state<Errored>,

        sml::state<Errored> + sml::event<event::render>[guard::valid_render] /
                                  action::begin_render = sml::state<Setup>,
        sml::state<Errored> + sml::event<event::render>[guard::invalid_render] /
                                  action::reject_invalid_render =
            sml::state<Errored>,

        sml::state<Unexpected> +
            sml::event<event::render>[guard::valid_render] / action::begin_render =
                sml::state<Setup>,
        sml::state<Unexpected> +
            sml::event<event::render>[guard::invalid_render] /
                action::reject_invalid_render = sml::state<Unexpected>,

        sml::state<Setup> / action::seed_program = sml::state<EvalStmt>,

        sml::state<EvalStmt>[guard::phase_failed{}] = sml::state<RenderDecision>,
        sml::state<EvalStmt>[guard::needs_expr{}] = sml::state<EvalExpr>,
        sml::state<EvalStmt>[guard::has_stmt_work{}] / action::eval_next_stmt =
            sml::state<EvalStmt>,
        sml::state<EvalStmt>[guard::no_stmt_work{}] = sml::state<RenderDecision>,

        sml::state<EvalExpr>[guard::has_pending_expr{}] / action::eval_pending_expr =
            sml::state<EvalExpr>,
        sml::state<EvalExpr>[guard::phase_failed{}] = sml::state<RenderDecision>,
        sml::state<EvalExpr> = sml::state<WriteOutput>,

        sml::state<WriteOutput>[guard::phase_failed{}] = sml::state<RenderDecision>,
        sml::state<WriteOutput>[guard::needs_write{}] / action::write_pending_value =
            sml::state<WriteOutput>,
        sml::state<WriteOutput> = sml::state<EvalStmt>,

        sml::state<RenderDecision>[guard::phase_ok{}] / action::finalize_done =
            sml::state<Done>,
        sml::state<RenderDecision>[guard::phase_failed{}] / action::finalize_error =
            sml::state<Errored>,

        sml::state<Initialized> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>,
        sml::state<Setup> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>,
        sml::state<EvalStmt> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>,
        sml::state<EvalExpr> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>,
        sml::state<WriteOutput> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>,
        sml::state<RenderDecision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>,
        sml::state<Done> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>,
        sml::state<Errored> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>,
        sml::state<Unexpected> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<Unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;
  using base_type::base_type;
  using base_type::process_event;
};

}  // namespace emel::jinja::renderer
