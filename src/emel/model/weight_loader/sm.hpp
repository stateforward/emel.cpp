#pragma once

#include "boost/sml.hpp"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::weight_loader {

struct initialized {};
struct loading_mmap {};
struct loading_streamed {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::load_weights>[guard::use_mmap{}]
        / action::load_mmap{} = sml::state<loading_mmap>,
      sml::state<initialized> + sml::event<event::load_weights>[guard::use_stream{}]
        / action::load_streamed{} = sml::state<loading_streamed>,

      sml::state<loading_mmap> + sml::event<events::weights_loaded>[guard::no_error{}]
        / action::store_and_dispatch_done{} = sml::state<done>,
      sml::state<loading_mmap> + sml::event<events::weights_loaded>[guard::has_error{}]
        / action::store_and_dispatch_error{} = sml::state<errored>,

      sml::state<loading_streamed> + sml::event<events::weights_loaded>[guard::no_error{}]
        / action::store_and_dispatch_done{} = sml::state<done>,
      sml::state<loading_streamed> + sml::event<events::weights_loaded>[guard::has_error{}]
        / action::store_and_dispatch_error{} = sml::state<errored>,

      sml::state<initialized> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<loading_mmap> + sml::event<sml::_> / action::on_unexpected{} =
        sml::state<errored>,
      sml::state<loading_streamed> + sml::event<sml::_> / action::on_unexpected{} =
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
