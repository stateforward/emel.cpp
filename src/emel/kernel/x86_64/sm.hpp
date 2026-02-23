#pragma once

// benchmark: scaffold
// docs: disabled

#include "emel/sm.hpp"
#include "emel/kernel/x86_64/events.hpp"

namespace emel::kernel::x86_64 {

struct idle {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::scaffold> = sml::state<idle>,
      sml::state<idle> + sml::unexpected_event<sml::_> = sml::state<idle>
    );
  }
};

using sm = emel::sm<model>;

}  // namespace emel::kernel::x86_64
