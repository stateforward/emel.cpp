#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/codec/encoder/actions.hpp"
#include "emel/codec/encoder/events.hpp"
#include "emel/codec/encoder/guards.hpp"

namespace emel::codec::encoder {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct tokenizing {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::encode> = sml::state<tokenizing>,
      sml::state<tokenizing> + sml::event<event::tokenized_done> = sml::state<done>,
      sml::state<tokenizing> + sml::event<event::tokenized_error> = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;

 private:
  int32_t status_code = 0;
  int32_t n_tokens = 0;
};

}  // namespace emel::codec::encoder
