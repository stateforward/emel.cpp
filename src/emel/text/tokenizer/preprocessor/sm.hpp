#pragma once

// benchmark: scaffold

#include "emel/sm.hpp"
#include "emel/text/tokenizer/preprocessor/events.hpp"

namespace emel::text::tokenizer::preprocessor {

struct idle {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::preprocess> = sml::state<idle>,
      sml::state<idle> + sml::unexpected_event<sml::_> = sml::state<idle>
    );
  }
};

using sm = emel::sm<model>;

}  // namespace emel::text::tokenizer::preprocessor
