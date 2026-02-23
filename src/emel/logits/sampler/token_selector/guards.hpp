#pragma once

#include "emel/logits/sampler/token_selector/actions.hpp"

namespace emel::logits::sampler::token_selector::guard {

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

}  // namespace emel::logits::sampler::token_selector::guard
