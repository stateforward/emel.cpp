#pragma once

#include "emel/sm.hpp"
#include "emel/model/loader/actions.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/guards.hpp"
#include "emel/model/parser/events.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::loader {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct mapping_parser {};
    struct parsing {};
    struct loading_weights {};
    struct mapping_layers {};
    struct validating_structure {};
    struct validating_architecture {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::load> = sml::state<mapping_parser>,

      sml::state<mapping_parser> + sml::event<event::mapping_parser_done> = sml::state<parsing>,
      sml::state<mapping_parser> + sml::event<event::unsupported_format_error> = sml::state<errored>,

      sml::state<parsing> + sml::event<emel::model::parser::events::parsing_done> =
        sml::state<loading_weights>,
      sml::state<parsing> + sml::event<emel::model::parser::events::parsing_error> =
        sml::state<errored>,

      sml::state<loading_weights> + sml::event<emel::model::weight_loader::events::loading_done> =
        sml::state<mapping_layers>,
      sml::state<loading_weights> + sml::event<emel::model::weight_loader::events::loading_error> =
        sml::state<errored>,

      sml::state<mapping_layers> + sml::event<event::layers_mapped>[guard::no_error] =
        sml::state<validating_structure>,
      sml::state<mapping_layers> + sml::event<event::layers_mapped>[guard::has_error] =
        sml::state<errored>,

      sml::state<validating_structure> +
          sml::event<event::structure_validated>[guard::no_error_and_has_arch_validate] =
        sml::state<validating_architecture>,
      sml::state<validating_structure> +
          sml::event<event::structure_validated>[guard::no_error_and_no_arch_validate] =
        sml::state<done>,
      sml::state<validating_structure> + sml::event<event::structure_validated>[guard::has_error] =
        sml::state<errored>,

      sml::state<validating_architecture> +
          sml::event<event::architecture_validated>[guard::no_error] = sml::state<done>,
      sml::state<validating_architecture> +
          sml::event<event::architecture_validated>[guard::has_error] = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;
};

inline bool load(sm & state_machine, const event::load & ev) {
  return state_machine.process_event(ev);
}

}  // namespace emel::model::loader
