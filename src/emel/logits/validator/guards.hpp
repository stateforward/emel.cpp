#pragma once

#include "emel/logits/validator/actions.hpp"

namespace emel::logits::validator::guard {

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

}  // namespace emel::logits::validator::guard
