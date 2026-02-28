#pragma once

#include "emel/text/tokenizer/preprocessor/context.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"
#include "emel/text/tokenizer/preprocessor/events.hpp"
#include "emel/text/tokenizer/preprocessor/types.hpp"

namespace emel::text::tokenizer::preprocessor::guard {

struct valid_request {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    if (ev.request.fragments_out.data() == nullptr) {
      return false;
    }
    if (ev.request.fragments_out.empty() ||
        ev.request.fragments_out.size() > k_max_fragments) {
      return false;
    }
    return true;
  }
};

struct invalid_request {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !valid_request{}(runtime_ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return preprocessor::is_ok(ev.ctx.phase_error);
  }
};

struct phase_failed {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return !preprocessor::is_ok(ev.ctx.phase_error);
  }
};

struct has_specials {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.special_cache.count != 0;
  }
};

struct no_specials {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.special_cache.count == 0;
  }
};

}  // namespace emel::text::tokenizer::preprocessor::guard
