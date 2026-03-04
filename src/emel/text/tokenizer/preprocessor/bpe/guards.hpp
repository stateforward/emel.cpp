#pragma once

#include "emel/text/tokenizer/preprocessor/guards.hpp"

namespace emel::text::tokenizer::preprocessor::bpe::guard {

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

struct phase_ok {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context & ctx) const noexcept {
    return emel::text::tokenizer::preprocessor::guard::phase_ok{}(runtime_ev, ctx);
  }
};

struct phase_failed {
  bool operator()(const event::preprocess_runtime & runtime_ev,
                  const action::context & ctx) const noexcept {
    return emel::text::tokenizer::preprocessor::guard::phase_failed{}(runtime_ev, ctx);
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

}  // namespace emel::text::tokenizer::preprocessor::bpe::guard
