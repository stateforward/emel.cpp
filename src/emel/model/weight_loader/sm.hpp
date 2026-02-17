#pragma once

#include "boost/sml.hpp"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::weight_loader {

struct initialized {};
struct selecting {};
struct initializing {};
struct loading_mmap {};
struct loading_streamed {};
struct validating {};
struct cleaning_up {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::load_weights> / action::select_strategy{} =
        sml::state<selecting>,

      sml::state<selecting> + sml::event<events::strategy_selected>
        [guard::use_mmap_no_error_can_init_mappings{}] / action::init_mappings{} =
          sml::state<initializing>,
      sml::state<selecting> + sml::event<events::strategy_selected>
        [guard::use_mmap_no_error_skip_init_mappings{}] / action::skip_init_mappings{} =
          sml::state<initializing>,
      sml::state<selecting> + sml::event<events::strategy_selected>
        [guard::use_mmap_no_error_cannot_init_mappings{}] / action::reject_invalid_mappings{} =
          sml::state<initializing>,
      sml::state<selecting> + sml::event<events::strategy_selected>
        [guard::use_stream_no_error_can_load_streamed{}] / action::load_streamed{} =
          sml::state<loading_streamed>,
      sml::state<selecting> + sml::event<events::strategy_selected>
        [guard::use_stream_no_error_cannot_load_streamed{}] / action::reject_invalid_streamed{} =
          sml::state<loading_streamed>,
      sml::state<selecting> + sml::event<events::strategy_selected>[guard::has_error{}]
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<initializing> + sml::event<events::mappings_ready>
        [guard::mappings_ready_no_error_can_load_mmap{}] / action::load_mmap{} =
          sml::state<loading_mmap>,
      sml::state<initializing> + sml::event<events::mappings_ready>
        [guard::mappings_ready_no_error_cannot_load_mmap{}] / action::reject_invalid_mmap{} =
          sml::state<loading_mmap>,
      sml::state<initializing> + sml::event<events::mappings_ready>[guard::has_error{}]
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<loading_mmap> + sml::event<events::weights_loaded>
        [guard::weights_loaded_no_error_can_validate{}] / action::store_and_validate{} =
          sml::state<validating>,
      sml::state<loading_mmap> + sml::event<events::weights_loaded>
        [guard::weights_loaded_no_error_skip_validate{}] / action::store_and_skip_validate{} =
          sml::state<validating>,
      sml::state<loading_mmap> + sml::event<events::weights_loaded>
        [guard::weights_loaded_no_error_cannot_validate{}]
          / action::store_and_reject_validate{} = sml::state<validating>,
      sml::state<loading_mmap> + sml::event<events::weights_loaded>[guard::has_error{}]
        / action::store_and_dispatch_error{} = sml::state<errored>,

      sml::state<loading_streamed> + sml::event<events::weights_loaded>
        [guard::weights_loaded_no_error_can_validate{}] / action::store_and_validate{} =
          sml::state<validating>,
      sml::state<loading_streamed> + sml::event<events::weights_loaded>
        [guard::weights_loaded_no_error_skip_validate{}] / action::store_and_skip_validate{} =
          sml::state<validating>,
      sml::state<loading_streamed> + sml::event<events::weights_loaded>
        [guard::weights_loaded_no_error_cannot_validate{}]
          / action::store_and_reject_validate{} = sml::state<validating>,
      sml::state<loading_streamed> + sml::event<events::weights_loaded>[guard::has_error{}]
        / action::store_and_dispatch_error{} = sml::state<errored>,

      sml::state<validating> + sml::event<events::validation_done>
        [guard::validation_done_no_error_can_clean_up{}] / action::cleaning_up{} =
          sml::state<cleaning_up>,
      sml::state<validating> + sml::event<events::validation_done>
        [guard::validation_done_no_error_skip_clean_up{}] / action::skip_cleaning_up{} =
          sml::state<cleaning_up>,
      sml::state<validating> + sml::event<events::validation_done>
        [guard::validation_done_no_error_cannot_clean_up{}] / action::reject_invalid_cleaning{} =
          sml::state<cleaning_up>,
      sml::state<validating> + sml::event<events::validation_done>[guard::has_error{}]
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<cleaning_up> + sml::event<events::cleaning_up_done>[guard::no_error{}]
        / action::dispatch_done{} = sml::state<done>,
      sml::state<cleaning_up> + sml::event<events::cleaning_up_done>[guard::has_error{}]
        / action::dispatch_error{} = sml::state<errored>,

      sml::state<initialized> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<selecting> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<initializing> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<loading_mmap> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<loading_streamed> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<validating> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<cleaning_up> + sml::event<sml::_> / action::on_unexpected{} =
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

inline bool dispatch_load_weights(void * loader_sm, const event::load_weights & ev) {
  auto * machine = static_cast<sm *>(loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

}  // namespace emel::model::weight_loader
