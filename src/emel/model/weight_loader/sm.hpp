#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/guards.hpp"

namespace emel::model::weight_loader {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct loading_mmap {};
    struct loading_streamed {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::load_weights>[guard::use_mmap] =
        sml::state<loading_mmap>,
      sml::state<initialized> + sml::event<event::load_weights>[guard::not_use_mmap] =
        sml::state<loading_streamed>,

      sml::state<loading_mmap> + sml::event<event::weights_loaded>[guard::no_error] = sml::state<done>,
      sml::state<loading_mmap> + sml::event<event::weights_loaded>[guard::has_error] = sml::state<errored>,

      sml::state<loading_streamed> + sml::event<event::weights_loaded>[guard::no_error] =
        sml::state<done>,
      sml::state<loading_streamed> + sml::event<event::weights_loaded>[guard::has_error] =
        sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;

  bool load(const event::load_weights & ev) {
    return process_event(ev);
  }

 private:
  int32_t status_code = 0;
  int32_t n_weight_files = 0;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
};

}  // namespace emel::model::weight_loader
