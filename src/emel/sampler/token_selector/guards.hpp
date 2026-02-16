#pragma once

#include "emel/sampler/token_selector/actions.hpp"

namespace emel::sampler::token_selector::guard {

struct has_candidates {
  template <class TEvent>
  bool operator()(const TEvent &, const action::context & ctx) const noexcept {
    return ctx.candidate_count > 0;
  }
};

struct no_candidates {
  template <class TEvent>
  bool operator()(const TEvent &, const action::context & ctx) const noexcept {
    return ctx.candidate_count <= 0;
  }
};

}  // namespace emel::sampler::token_selector::guard
