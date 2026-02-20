#pragma once

#include <cstdint>
#include <type_traits>

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/parser/actions.hpp"
#include "emel/parser/events.hpp"
#include "emel/parser/guards.hpp"
#include "emel/parser/gguf/actions.hpp"
#include "emel/sm.hpp"

namespace emel::parser::gguf {

struct initialized {};
struct parsing_architecture {};
struct parse_architecture_decision {};
struct mapping_architecture {};
struct map_architecture_decision {};
struct parsing_hparams {};
struct parse_hparams_decision {};
struct parsing_vocab {};
struct parse_vocab_decision {};
struct mapping_tensors {};
struct map_tensors_decision {};
struct done {};
struct errored {};

/**
 * GGUF parser orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting parse intent.
 * - `parsing_*`: run parser steps.
 * - `*_decision`: branch on phase error or configuration.
 * - `done`/`errored`: terminal outcomes that return to initialized.
 *
 * Guard semantics:
 * - `valid_parse_request` validates required request fields.
 * - `phase_*` guards observe `context.phase_error`.
 *
 * Action side effects:
 * - Actions execute bounded parsing hooks and set `phase_error`.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::parse_model>
        [guard::valid_parse_request{}] / parser::action::begin_parse =
          sml::state<parsing_architecture>,
      sml::state<initialized> + sml::event<event::parse_model>
        [guard::invalid_parse_request{}] / parser::action::set_invalid_argument =
          sml::state<errored>,

      sml::state<parsing_architecture> / gguf::action::run_parse_architecture =
          sml::state<parse_architecture_decision>,
      sml::state<parse_architecture_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<parse_architecture_decision> [guard::phase_ok{}] =
          sml::state<mapping_architecture>,

      sml::state<mapping_architecture> / gguf::action::run_map_architecture =
          sml::state<map_architecture_decision>,
      sml::state<map_architecture_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<map_architecture_decision> [guard::phase_ok{}] =
          sml::state<parsing_hparams>,

      sml::state<parsing_hparams> / gguf::action::run_parse_hparams =
          sml::state<parse_hparams_decision>,
      sml::state<parse_hparams_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<parse_hparams_decision> [guard::phase_ok{}] =
          sml::state<parsing_vocab>,

      sml::state<parsing_vocab> / gguf::action::run_parse_vocab =
          sml::state<parse_vocab_decision>,
      sml::state<parse_vocab_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<parse_vocab_decision> [guard::phase_ok_and_skip_map_tensors{}] =
          sml::state<done>,
      sml::state<parse_vocab_decision> [guard::phase_ok_and_map_tensors{}] =
          sml::state<mapping_tensors>,

      sml::state<mapping_tensors> / gguf::action::run_map_tensors =
          sml::state<map_tensors_decision>,
      sml::state<map_tensors_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<map_tensors_decision> [guard::phase_ok{}] =
          sml::state<done>,

      sml::state<done> / parser::action::mark_done = sml::state<initialized>,
      sml::state<errored> / parser::action::ensure_last_error = sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<parsing_architecture> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<parse_architecture_decision> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<mapping_architecture> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<map_architecture_decision> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<parsing_hparams> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<parse_hparams_decision> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<parsing_vocab> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<parse_vocab_decision> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<mapping_tensors> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<map_tensors_decision> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> /
          parser::action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::parse_model & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, emel::model::loader::events::parsing_done{ev.loader_request});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, emel::model::loader::events::parsing_error{
          ev.loader_request,
          err,
        });
      }
    }
    parser::action::clear_request_action(context_);
    return accepted && err == EMEL_OK;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  using base_type::raw_sm;

  parser::action::context context_{};
};

}  // namespace emel::parser::gguf
