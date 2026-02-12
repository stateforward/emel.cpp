#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/codec/decoder/actions.hpp"
#include "emel/codec/decoder/events.hpp"
#include "emel/codec/decoder/guards.hpp"

namespace emel::codec::decoder {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct detokenizing {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::decode> = sml::state<detokenizing>,
      sml::state<detokenizing> + sml::event<event::detokenized_done> = sml::state<done>,
      sml::state<detokenizing> + sml::event<event::detokenized_error> = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;

 private:
  int32_t status_code = 0;
};

}  // namespace emel::codec::decoder
