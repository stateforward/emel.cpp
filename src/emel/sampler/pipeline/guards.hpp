#pragma once

#include "emel/sampler/pipeline/actions.hpp"

namespace emel::sampler::pipeline::guard {

struct has_more_samplers {
  template <class TEvent>
  bool operator()(const TEvent &, const action::context & ctx) const noexcept {
    return ctx.sampler_index < ctx.sampler_count;
  }
};

struct no_more_samplers {
  template <class TEvent>
  bool operator()(const TEvent &, const action::context & ctx) const noexcept {
    return ctx.sampler_index >= ctx.sampler_count;
  }
};

}  // namespace emel::sampler::pipeline::guard
