#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/tokenizer/actions.hpp"
#include "emel/tokenizer/events.hpp"
#include "emel/tokenizer/guards.hpp"

namespace emel::tokenizer {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct tokenizing {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::tokenize>[guard::can_tokenize] /
          action::on_tokenize_requested = sml::state<tokenizing>,
      sml::state<tokenizing> + sml::event<event::tokenizing_done> / action::on_tokenizing_done =
          sml::state<done>,
      sml::state<tokenizing> + sml::event<event::tokenizing_error> /
          action::on_tokenizing_error = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;

 private:
  int32_t status_code = 0;
  int32_t n_tokens = 0;
};

}  // namespace emel::tokenizer
