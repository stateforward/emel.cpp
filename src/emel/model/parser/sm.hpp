#pragma once

#include "emel/sm.hpp"
#include "emel/model/parser/actions.hpp"
#include "emel/model/parser/events.hpp"
#include "emel/model/parser/guards.hpp"

namespace emel::model::parser {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct parsing_architecture {};
    struct mapping_architecture {};
    struct parsing_hparams {};
    struct parsing_vocab {};
    struct mapping_tensors {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::parse_model> = sml::state<parsing_architecture>,

      sml::state<parsing_architecture> + sml::event<event::parse_architecture_done> =
        sml::state<mapping_architecture>,
      sml::state<parsing_architecture> + sml::event<event::parse_architecture_error> =
        sml::state<errored>,

      sml::state<mapping_architecture> + sml::event<event::map_architecture_done> =
        sml::state<parsing_hparams>,
      sml::state<mapping_architecture> + sml::event<event::map_architecture_error> =
        sml::state<errored>,

      sml::state<parsing_hparams> + sml::event<event::parse_hparams_done> = sml::state<parsing_vocab>,
      sml::state<parsing_hparams> + sml::event<event::parse_hparams_error> = sml::state<errored>,

      sml::state<parsing_vocab> + sml::event<event::parse_vocab_done> = sml::state<mapping_tensors>,
      sml::state<parsing_vocab> + sml::event<event::parse_vocab_error> = sml::state<errored>,

      sml::state<mapping_tensors> + sml::event<event::map_tensors_done> = sml::state<done>,
      sml::state<mapping_tensors> + sml::event<event::map_tensors_error> = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;
};

}  // namespace emel::model::parser
