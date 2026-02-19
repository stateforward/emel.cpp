#pragma once

#include "emel/sampler/candidate_builder/actions.hpp"

namespace emel::sampler::candidate_builder::guard {

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

}  // namespace emel::sampler::candidate_builder::guard
