#pragma once

#include <cstdint>

#include "emel/jinja/renderer/actions.hpp"
#include "emel/jinja/renderer/events.hpp"
#include "emel/jinja/renderer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::jinja::renderer {

struct Initialized {};
struct RenderDecision {};
struct Done {};
struct Errored {};
struct Unexpected {};

/**
 * Jinja renderer orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting render intent.
 * - `render_decision`: run render step and branch based on phase results.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * Guard semantics:
 * - `valid_render`/`invalid_render` validate request pointers and parameters.
 * - `phase_*` guards observe errors set by actions.
 *
 * Action side effects:
 * - `run_render` executes rendering and dispatches callbacks.
 * - `reject_invalid_render` writes errors for invalid requests.
 * - `on_unexpected` reports sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<Initialized> +
            sml::event<event::render>[guard::valid_render] / action::run_render =
                sml::state<RenderDecision>,
        sml::state<Initialized> +
            sml::event<event::render>[guard::invalid_render] /
                action::reject_invalid_render = sml::state<Errored>,

        sml::state<Done> + sml::event<event::render>[guard::valid_render] /
                               action::run_render = sml::state<RenderDecision>,
        sml::state<Done> + sml::event<event::render>[guard::invalid_render] /
                               action::reject_invalid_render =
            sml::state<Errored>,

        sml::state<Errored> + sml::event<event::render>[guard::valid_render] /
                                  action::run_render = sml::state<RenderDecision>,
        sml::state<Errored> + sml::event<event::render>[guard::invalid_render] /
                                  action::reject_invalid_render =
            sml::state<Errored>,

        sml::state<Unexpected> +
            sml::event<event::render>[guard::valid_render] / action::run_render =
                sml::state<RenderDecision>,
        sml::state<Unexpected> +
            sml::event<event::render>[guard::invalid_render] /
                action::reject_invalid_render = sml::state<Unexpected>,

        sml::state<RenderDecision>[guard::phase_ok{}] = sml::state<Done>,
        sml::state<RenderDecision>[guard::phase_failed{}] = sml::state<Errored>,

        sml::state<Initialized> + sml::unexpected_event<sml::_> /
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
