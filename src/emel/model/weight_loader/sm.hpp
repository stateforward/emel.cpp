#pragma once

#include "boost/sml.hpp"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::weight_loader {

struct initialized {};
struct selecting {};
struct strategy_decision {};
struct initializing {};
struct init_decision {};
struct loading_mmap {};
struct loading_streamed {};
struct load_decision {};
struct validating {};
struct validation_decision {};
struct cleanup_decision {};
struct cleaning_up {};
struct cleanup_result {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::load_weights> / action::begin_load =
        sml::state<selecting>,

      sml::state<selecting> / action::select_strategy = sml::state<strategy_decision>,
      sml::state<strategy_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<strategy_decision> [guard::phase_ok_and_use_mmap_and_can_init_mappings{}] =
        sml::state<initializing>,
      sml::state<strategy_decision> [guard::phase_ok_and_use_mmap_and_skip_init_mappings{}] =
        sml::state<loading_mmap>,
      sml::state<strategy_decision> [guard::phase_ok_and_use_mmap_and_cannot_init_mappings{}]
        / action::set_invalid_argument = sml::state<errored>,
      sml::state<strategy_decision> [guard::phase_ok_and_use_stream_and_can_load_streamed{}] =
        sml::state<loading_streamed>,
      sml::state<strategy_decision> [guard::phase_ok_and_use_stream_and_cannot_load_streamed{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<initializing> / action::run_init_mappings = sml::state<init_decision>,
      sml::state<init_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<init_decision> [guard::phase_ok_and_can_load_mmap{}] =
        sml::state<loading_mmap>,
      sml::state<init_decision> [guard::phase_ok_and_cannot_load_mmap{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<loading_mmap> / action::run_load_mmap = sml::state<load_decision>,
      sml::state<loading_streamed> / action::run_load_streamed = sml::state<load_decision>,
      sml::state<load_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<load_decision> [guard::phase_ok_and_can_validate{}] =
        sml::state<validating>,
      sml::state<load_decision> [guard::phase_ok_and_skip_validate{}] =
        sml::state<cleanup_decision>,
      sml::state<load_decision> [guard::phase_ok_and_cannot_validate{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<validating> / action::run_validate = sml::state<validation_decision>,
      sml::state<validation_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<validation_decision> [guard::phase_ok{}] = sml::state<cleanup_decision>,

      sml::state<cleanup_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<cleanup_decision> [guard::phase_ok_and_can_clean_up{}] =
        sml::state<cleaning_up>,
      sml::state<cleanup_decision> [guard::phase_ok_and_skip_clean_up{}] =
        sml::state<done>,
      sml::state<cleanup_decision> [guard::phase_ok_and_cannot_clean_up{}]
        / action::set_invalid_argument = sml::state<errored>,

      sml::state<cleaning_up> / action::run_clean_up = sml::state<cleanup_result>,
      sml::state<cleanup_result> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<cleanup_result> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::publish_done = sml::state<initialized>,
      sml::state<errored> / action::publish_error = sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<selecting> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<strategy_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<initializing> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<init_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<loading_mmap> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<loading_streamed> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<load_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<validating> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<validation_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<cleanup_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<cleaning_up> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<cleanup_result> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> /
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

inline bool dispatch_load_weights(void * loader_sm, const event::load_weights & ev) {
  auto * machine = static_cast<sm *>(loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

}  // namespace emel::model::weight_loader
