#pragma once

// benchmark: scaffold
// docs: disabled

#include "emel/sm.hpp"
#include "emel/gbnf/lexer/events.hpp"

namespace emel::gbnf::lexer {

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

}  // namespace emel::gbnf::lexer
