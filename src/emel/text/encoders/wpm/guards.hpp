#pragma once

#include <cstddef>

#include "emel/text/encoders/wpm/context.hpp"
#include "emel/text/encoders/guards.hpp"

namespace emel::text::encoders::wpm::guard {

struct valid_encode {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::valid_encode{}(ev, ctx);
  }
};

struct invalid_encode {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::invalid_encode{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::phase_ok{}(ev);
  }
};

struct phase_failed {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::phase_failed{}(ev);
  }
};

struct text_empty {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::text_empty{}(ev);
  }
};

struct text_non_empty {
  bool operator()(const event::encode_runtime & ev) const noexcept {
    return emel::text::encoders::guard::text_non_empty{}(ev);
  }
};

struct prefix_buffer_capacity_within_limit {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    constexpr size_t k_wpm_prefix_len = 3u;
    const bool has_prefix_capacity = ctx.scratch.buffer.size() >= k_wpm_prefix_len;
    const size_t max_word_bytes =
      ctx.scratch.buffer.size() - (k_wpm_prefix_len * static_cast<size_t>(has_prefix_capacity));
    return has_prefix_capacity && ev.request.text.size() <= max_word_bytes;
  }
};

struct prefix_buffer_capacity_exceeded {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !prefix_buffer_capacity_within_limit{}(ev, ctx);
  }
};

struct vocab_changed {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::vocab_changed{}(ev, ctx);
  }
};

struct vocab_unchanged {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return emel::text::encoders::guard::vocab_unchanged{}(ev, ctx);
  }
};

struct tables_ready {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    (void)ev;
    return ctx.tables_ready && ctx.vocab == &ev.request.vocab;
  }
};

struct tables_missing {
  bool operator()(const event::encode_runtime & ev, const action::context & ctx) const noexcept {
    return !tables_ready{}(ev, ctx);
  }
};

}  // namespace emel::text::encoders::wpm::guard
