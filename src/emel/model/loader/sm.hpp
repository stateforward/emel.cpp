#pragma once

#include "boost/sml.hpp"
#include "emel/model/loader/actions.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::loader {

struct initialized {};
struct mapping_parser {};
struct map_parser_decision {};
struct parsing {};
struct parse_decision {};
struct loading_weights {};
struct load_decision {};
struct mapping_layers {};
struct map_layers_decision {};
struct validating_structure {};
struct structure_skipped {};
struct structure_decision {};
struct validating_architecture {};
struct architecture_decision {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    const auto not_anonymous = [](const auto & ev) {
      using event_type = std::decay_t<decltype(ev)>;
      return !std::is_same_v<event_type, boost::sml::anonymous>;
    };
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::load> [guard::can_map_parser{}]
        / action::begin_load = sml::state<mapping_parser>,
      sml::state<initialized> + sml::event<event::load> [guard::cannot_map_parser{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<mapping_parser> / action::run_map_parser = sml::state<map_parser_decision>,
      sml::state<map_parser_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<map_parser_decision> [guard::phase_ok_and_can_parse{}] =
        sml::state<parsing>,
      sml::state<map_parser_decision> [guard::phase_ok_and_cannot_parse{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<parsing> / action::run_parse = sml::state<parse_decision>,
      sml::state<parse_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<parse_decision> [guard::phase_ok_and_should_load_weights_and_can_load{}] =
        sml::state<loading_weights>,
      sml::state<parse_decision> [guard::phase_ok_and_should_load_weights_and_cannot_load{}]
        / action::set_invalid_argument = sml::state<errored>,
      sml::state<parse_decision> [guard::phase_ok_and_skip_weights_and_skip_structure{}] =
        sml::state<structure_skipped>,
      sml::state<parse_decision> [guard::phase_ok_and_skip_weights_and_can_validate_structure{}] =
        sml::state<validating_structure>,
      sml::state<parse_decision> [guard::phase_ok_and_skip_weights_and_cannot_validate_structure{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<loading_weights> / action::run_load_weights = sml::state<load_decision>,
      sml::state<load_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<load_decision> [guard::phase_ok_and_can_map_layers{}] =
        sml::state<mapping_layers>,
      sml::state<load_decision> [guard::phase_ok_and_cannot_map_layers{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<mapping_layers> / action::run_map_layers = sml::state<map_layers_decision>,
      sml::state<map_layers_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<map_layers_decision> [guard::phase_ok_and_skip_structure{}] =
        sml::state<structure_skipped>,
      sml::state<map_layers_decision> [guard::phase_ok_and_can_validate_structure{}] =
        sml::state<validating_structure>,
      sml::state<map_layers_decision> [guard::phase_ok_and_cannot_validate_structure{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<validating_structure> / action::run_validate_structure =
        sml::state<structure_decision>,
      sml::state<structure_skipped> / action::skip_validate_structure =
        sml::state<structure_decision>,
      sml::state<structure_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<structure_decision>
        [guard::phase_ok_and_has_arch_validate_and_can_validate_architecture{}] =
          sml::state<validating_architecture>,
      sml::state<structure_decision>
        [guard::phase_ok_and_has_arch_validate_and_cannot_validate_architecture{}]
          / action::set_invalid_argument = sml::state<errored>,
      sml::state<structure_decision> [guard::phase_ok_and_no_arch_validate{}] =
        sml::state<done>,

      sml::state<validating_architecture> / action::run_validate_architecture =
        sml::state<architecture_decision>,
      sml::state<architecture_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<architecture_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::publish_done = sml::state<initialized>,
      sml::state<errored> / action::publish_error = sml::state<initialized>,

      sml::state<initialized> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<mapping_parser> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<map_parser_decision> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<parsing> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<parse_decision> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<loading_weights> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<load_decision> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<mapping_layers> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<map_layers_decision> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<validating_structure> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<structure_skipped> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<structure_decision> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<validating_architecture> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<architecture_decision> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;
  using base_type::visit_current_states;

 private:
  action::context context_{};
};

}  // namespace emel::model::loader
