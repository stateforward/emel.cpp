#pragma once

#include "boost/sml.hpp"
#include "emel/model/loader/actions.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::loader {

struct initialized {};
struct mapping_parser {};
struct parsing {};
struct loading_weights {};
struct mapping_layers {};
struct validating_structure {};
struct validating_architecture {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::load> [guard::can_map_parser{}]
        / action::start_map_parser{} = sml::state<mapping_parser>,
      sml::state<initialized> + sml::event<event::load> [guard::cannot_map_parser{}]
        / action::reject_invalid{} = sml::state<errored>,

      sml::state<mapping_parser> + sml::event<events::mapping_parser_done> [guard::can_parse{}]
        / action::parse{} = sml::state<parsing>,
      sml::state<mapping_parser> + sml::event<events::mapping_parser_done> [guard::cannot_parse{}]
        / action::reject_invalid{} = sml::state<errored>,
      sml::state<mapping_parser> + sml::event<events::mapping_parser_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<parsing> + sml::event<events::parsing_done>
        [guard::should_load_weights_and_can_load{}]
        / action::load_weights{} = sml::state<loading_weights>,
      sml::state<parsing> + sml::event<events::parsing_done>
        [guard::should_load_weights_and_cannot_load{}]
        / action::reject_invalid{} = sml::state<errored>,
      sml::state<parsing> + sml::event<events::parsing_done>
        [guard::skip_weights_and_can_validate_structure{}]
        / action::validate_structure{} = sml::state<validating_structure>,
      sml::state<parsing> + sml::event<events::parsing_done>
        [guard::skip_weights_and_cannot_validate_structure{}]
        / action::reject_invalid_structure{} = sml::state<errored>,
      sml::state<parsing> + sml::event<events::parsing_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<loading_weights> + sml::event<events::loading_done> [guard::can_map_layers{}]
        / action::store_and_map_layers{} = sml::state<mapping_layers>,
      sml::state<loading_weights> + sml::event<events::loading_done> [guard::cannot_map_layers{}]
        / action::reject_invalid{} = sml::state<errored>,
      sml::state<loading_weights> + sml::event<events::loading_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<mapping_layers> + sml::event<events::layers_mapped>
        [guard::can_validate_structure{}] / action::validate_structure{}
          = sml::state<validating_structure>,
      sml::state<mapping_layers> + sml::event<events::layers_mapped>
        [guard::cannot_validate_structure{}] / action::reject_invalid_structure{}
          = sml::state<errored>,
      sml::state<mapping_layers> + sml::event<events::layers_map_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<validating_structure> + sml::event<events::structure_validated>
        [guard::has_arch_validate_and_can_validate_architecture{}]
          / action::validate_architecture{}
          = sml::state<validating_architecture>,
      sml::state<validating_structure> + sml::event<events::structure_validated>
        [guard::has_arch_validate_and_cannot_validate_architecture{}]
          / action::reject_invalid{} = sml::state<errored>,
      sml::state<validating_structure> + sml::event<events::structure_validated>
        [guard::no_arch_validate{}] / action::dispatch_done{} = sml::state<done>,
      sml::state<validating_structure> + sml::event<events::structure_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<validating_architecture> + sml::event<events::architecture_validated>
        / action::dispatch_done{} = sml::state<done>,
      sml::state<validating_architecture> + sml::event<events::architecture_error>
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<initialized> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<mapping_parser> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<parsing> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<loading_weights> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<mapping_layers> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<validating_structure> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<validating_architecture> + sml::event<sml::_> / action::on_unexpected{} =
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
        base_type(context_, this->process_) {}

  using base_type::process_event;

 private:
  action::context context_{};
};

}  // namespace emel::model::loader
