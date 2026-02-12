#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/buffer_allocator/actions.hpp"
#include "emel/buffer_allocator/events.hpp"
#include "emel/buffer_allocator/guards.hpp"

namespace emel::buffer_allocator {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct allocating {};
    struct uploading {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::allocate_buffer> = sml::state<allocating>,
      sml::state<allocating> + sml::event<events::allocation_done> = sml::state<uploading>,
      sml::state<allocating> + sml::event<events::allocation_error> = sml::state<errored>,
      sml::state<uploading> + sml::event<events::upload_done> = sml::state<done>,
      sml::state<uploading> + sml::event<events::upload_error> = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;

 private:
  int32_t status_code = 0;
};

}  // namespace emel::buffer_allocator
