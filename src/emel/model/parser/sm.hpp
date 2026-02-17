#pragma once

#include "boost/sml.hpp"
#include "emel/model/parser/actions.hpp"
#include "emel/model/parser/events.hpp"
#include "emel/model/parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::parser {

struct initialized {};
struct parsing_architecture {};
struct mapping_architecture {};
struct parsing_hparams {};
struct parsing_vocab {};
struct mapping_tensors {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::parse_model>
        [guard::can_parse_architecture{}] / action::parse_architecture{} =
          sml::state<parsing_architecture>,
      sml::state<initialized> + sml::event<event::parse_model>
        [guard::cannot_parse_architecture{}] / action::reject_invalid{} =
          sml::state<errored>,

      sml::state<parsing_architecture> + sml::event<events::parse_architecture_done>
        [guard::can_map_architecture{}] / action::map_architecture{} =
          sml::state<mapping_architecture>,
      sml::state<parsing_architecture> + sml::event<events::parse_architecture_done>
        [guard::cannot_map_architecture{}] / action::reject_invalid{} =
          sml::state<errored>,
      sml::state<parsing_architecture> + sml::event<events::parse_architecture_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<mapping_architecture> + sml::event<events::map_architecture_done>
        [guard::can_parse_hparams{}] / action::parse_hparams{} = sml::state<parsing_hparams>,
      sml::state<mapping_architecture> + sml::event<events::map_architecture_done>
        [guard::cannot_parse_hparams{}] / action::reject_invalid{} = sml::state<errored>,
      sml::state<mapping_architecture> + sml::event<events::map_architecture_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<parsing_hparams> + sml::event<events::parse_hparams_done>
        [guard::can_parse_vocab{}] / action::parse_vocab{} = sml::state<parsing_vocab>,
      sml::state<parsing_hparams> + sml::event<events::parse_hparams_done>
        [guard::cannot_parse_vocab{}] / action::reject_invalid{} = sml::state<errored>,
      sml::state<parsing_hparams> + sml::event<events::parse_hparams_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<parsing_vocab> + sml::event<events::parse_vocab_done>
        [guard::skip_map_tensors{}] / action::dispatch_done{} = sml::state<done>,
      sml::state<parsing_vocab> + sml::event<events::parse_vocab_done>
        [guard::can_map_tensors{}] / action::map_tensors{} = sml::state<mapping_tensors>,
      sml::state<parsing_vocab> + sml::event<events::parse_vocab_done>
        [guard::cannot_map_tensors{}] / action::reject_invalid{} = sml::state<errored>,
      sml::state<parsing_vocab> + sml::event<events::parse_vocab_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<mapping_tensors> + sml::event<events::map_tensors_done>
        / action::dispatch_done{} = sml::state<done>,
      sml::state<mapping_tensors> + sml::event<events::map_tensors_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<initialized> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<parsing_architecture> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<mapping_architecture> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<parsing_hparams> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<parsing_vocab> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<mapping_tensors> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<done> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<errored> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>
    );
  }
};

using Process = process_t;

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm()
      : emel::detail::process_support<sm, Process>(this),
        base_type(this->process_) {}

  using base_type::process_event;
};

inline bool dispatch_parse_model(void * parser_sm, const event::parse_model & ev) {
  auto * machine = static_cast<sm *>(parser_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

}  // namespace emel::model::parser
