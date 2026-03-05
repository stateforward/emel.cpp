#pragma once

#include "emel/text/tokenizer/preprocessor/guards.hpp"

namespace emel::text::tokenizer::preprocessor::plamo2::guard {

namespace pdetail = emel::text::tokenizer::preprocessor::detail;

struct fragments_buffer_present {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    return ev.request.fragments_out.data() != nullptr;
  }
};

struct fragments_buffer_missing {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !fragments_buffer_present{}(runtime_ev, ctx);
  }
};

struct fragments_capacity_nonzero {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    return !ev.request.fragments_out.empty();
  }
};

struct fragments_capacity_zero {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !fragments_capacity_nonzero{}(runtime_ev, ctx);
  }
};

struct fragments_capacity_within_limit {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    return ev.request.fragments_out.size() <= k_max_fragments;
  }
};

struct fragments_capacity_exceeds_limit {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !fragments_capacity_within_limit{}(runtime_ev, ctx);
  }
};

inline bool phase_error_is(const event::preprocess_runtime & runtime_ev,
                           const preprocessor::error err) noexcept {
  const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
  return ev.ctx.phase_error == err;
}

struct build_specials_ok {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    return phase_error_is(runtime_ev, preprocessor::error::none);
  }
};

struct build_specials_invalid_request_error {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    return phase_error_is(runtime_ev, preprocessor::error::invalid_request);
  }
};

struct build_specials_backend_error {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    return phase_error_is(runtime_ev, preprocessor::error::backend_error);
  }
};

struct build_specials_unknown_error {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.phase_error != preprocessor::error::none &&
           ev.ctx.phase_error != preprocessor::error::invalid_request &&
           ev.ctx.phase_error != preprocessor::error::backend_error;
  }
};

struct partition_ok {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    return phase_error_is(runtime_ev, preprocessor::error::none);
  }
};

struct partition_invalid_request_error {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    return phase_error_is(runtime_ev, preprocessor::error::invalid_request);
  }
};

struct partition_backend_error {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    return phase_error_is(runtime_ev, preprocessor::error::backend_error);
  }
};

struct partition_unknown_error {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.phase_error != preprocessor::error::none &&
           ev.ctx.phase_error != preprocessor::error::invalid_request &&
           ev.ctx.phase_error != preprocessor::error::backend_error;
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

struct parse_special_enabled {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    return ev.request.parse_special;
  }
};

struct parse_special_disabled {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !parse_special_enabled{}(runtime_ev, ctx);
  }
};

struct request_text_empty {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context &) const noexcept {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    return ev.request.text.empty();
  }
};

struct request_text_nonempty {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !request_text_empty{}(runtime_ev, ctx);
  }
};

}  // namespace emel::text::tokenizer::preprocessor::plamo2::guard
