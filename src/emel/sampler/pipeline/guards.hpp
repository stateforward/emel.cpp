#pragma once

#include "emel/sampler/pipeline/actions.hpp"

namespace emel::sampler::pipeline::guard {

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

struct has_more_samplers {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.sampler_index < ctx.sampler_count;
  }
};

struct no_more_samplers {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.sampler_index >= ctx.sampler_count;
  }
};

struct phase_ok_and_has_more_samplers {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && has_more_samplers{}(ctx);
  }
};

struct phase_ok_and_no_more_samplers {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && no_more_samplers{}(ctx);
  }
};

}  // namespace emel::sampler::pipeline::guard
