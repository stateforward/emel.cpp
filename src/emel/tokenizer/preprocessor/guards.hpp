#pragma once

#include "emel/tokenizer/preprocessor/context.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"

namespace emel::tokenizer::preprocessor::guard {

struct valid_request {
  bool operator()(const event::preprocess & ev, const action::context &) const noexcept {
    if (ev.vocab == nullptr) {
      return false;
    }
    if (ev.fragments_out == nullptr || ev.fragment_count_out == nullptr) {
      return false;
    }
    if (ev.error_out == nullptr) {
      return false;
    }
    if (ev.fragment_capacity == 0 ||
        ev.fragment_capacity > k_max_fragments) {
      return false;
    }
    return true;
  }
};

struct invalid_request {
  bool operator()(const event::preprocess & ev, const action::context & ctx) const noexcept {
    return !valid_request{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

}  // namespace emel::tokenizer::preprocessor::guard
