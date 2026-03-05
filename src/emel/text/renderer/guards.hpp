#pragma once

#include <string_view>

#include "emel/error/error.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/renderer/context.hpp"
#include "emel/text/renderer/errors.hpp"
#include "emel/text/renderer/events.hpp"

namespace emel::text::renderer::guard {

namespace detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

inline constexpr int32_t k_detokenizer_ok = static_cast<int32_t>(
    emel::error::cast(emel::text::detokenizer::error::none));
inline constexpr int32_t k_detokenizer_backend_error = static_cast<int32_t>(
    emel::error::cast(emel::text::detokenizer::error::backend_error));

inline bool is_leading_space(const char value) noexcept {
  return value == ' ' || value == '\t' || value == '\n' || value == '\r';
}

}  // namespace detail

struct valid_initialize {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    if constexpr (requires { ev.event_; }) {
      return false;
    }
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    if (runtime_ev.request.stop_sequence_count > 0 &&
        runtime_ev.request.stop_sequences == nullptr) {
      return false;
    }
    if (runtime_ev.request.stop_sequence_count > action::k_max_stop_sequences) {
      return false;
    }

    size_t used_storage = 0;
    for (size_t index = 0; index < runtime_ev.request.stop_sequence_count; ++index) {
      const std::string_view stop = runtime_ev.request.stop_sequences[index];
      if (stop.empty() || stop.size() > action::k_max_stop_length) {
        return false;
      }
      used_storage += stop.size();
      if (used_storage > action::k_max_stop_storage) {
        return false;
      }
    }

    return true;
  }
};

struct invalid_initialize {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return !valid_initialize{}(ev);
  }
};

struct valid_render {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    if constexpr (requires { ev.event_; }) {
      return false;
    }
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    if (ctx.vocab == nullptr) {
      return false;
    }
    if (runtime_ev.request.token_id < 0) {
      return false;
    }
    if (runtime_ev.request.sequence_id < 0 ||
        static_cast<size_t>(runtime_ev.request.sequence_id) >= action::k_max_sequences) {
      return false;
    }
    if (runtime_ev.request.output == nullptr &&
        runtime_ev.request.output_capacity > 0) {
      return false;
    }
    return true;
  }
};

struct invalid_render {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    return !valid_render{}(ev, ctx);
  }
};

struct valid_flush {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    if constexpr (requires { ev.event_; }) {
      return false;
    }
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    if (ctx.vocab == nullptr) {
      return false;
    }
    if (runtime_ev.request.sequence_id < 0 ||
        static_cast<size_t>(runtime_ev.request.sequence_id) >= action::k_max_sequences) {
      return false;
    }
    if (runtime_ev.request.output == nullptr &&
        runtime_ev.request.output_capacity > 0) {
      return false;
    }
    return true;
  }
};

struct invalid_flush {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    return !valid_flush{}(ev, ctx);
  }
};

struct request_ok {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == emel::error::cast(error::none);
  }
};

struct request_failed {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err != emel::error::cast(error::none);
  }
};

struct initialize_dispatch_backend_failure {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.detokenizer_err == detail::k_detokenizer_backend_error;
  }
};

struct initialize_dispatch_reported_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.detokenizer_err != detail::k_detokenizer_ok &&
           runtime_ev.ctx.detokenizer_err != detail::k_detokenizer_backend_error;
  }
};

struct initialize_dispatch_ok {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.detokenizer_err == detail::k_detokenizer_ok;
  }
};

struct sequence_stop_matched {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return ctx.sequences[runtime_ev.ctx.sequence_index].stop_matched;
  }
};

struct sequence_running {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    return !sequence_stop_matched{}(ev, ctx);
  }
};

struct render_dispatch_backend_failure {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.detokenizer_err == detail::k_detokenizer_backend_error;
  }
};

struct render_dispatch_reported_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.detokenizer_err != detail::k_detokenizer_ok &&
           runtime_ev.ctx.detokenizer_err != detail::k_detokenizer_backend_error;
  }
};

struct render_dispatch_lengths_valid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    const auto & sequence = ctx.sequences[runtime_ev.ctx.sequence_index];
    return runtime_ev.ctx.detokenizer_output_length <= runtime_ev.request.output_capacity &&
           runtime_ev.ctx.detokenizer_pending_length <= sequence.pending_bytes.size();
  }
};

struct render_dispatch_lengths_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    return !render_dispatch_lengths_valid{}(ev, ctx);
  }
};

struct render_dispatch_ok {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.detokenizer_err == detail::k_detokenizer_ok &&
           render_dispatch_lengths_valid{}(ev, ctx);
  }
};

struct strip_needed {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    const auto & sequence = ctx.sequences[runtime_ev.ctx.sequence_index];
    return render_dispatch_ok{}(ev, ctx) &&
           sequence.strip_leading_space &&
           runtime_ev.ctx.detokenizer_output_length > 0 &&
           detail::is_leading_space(runtime_ev.request.output[0]);
  }
};

struct strip_not_needed {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    return render_dispatch_ok{}(ev, ctx) && !strip_needed{}(ev, ctx);
  }
};

struct strip_prefix_nonzero {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.leading_space_prefix_length != 0;
  }
};

struct strip_prefix_zero {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.leading_space_prefix_length == 0;
  }
};

struct flush_output_fits {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    const auto & sequence = ctx.sequences[runtime_ev.ctx.sequence_index];
    const size_t required = sequence.pending_length + sequence.holdback_length;
    return required <= runtime_ev.request.output_capacity;
  }
};

struct flush_output_too_large {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev,
                  const action::context & ctx) const noexcept {
    return !flush_output_fits{}(ev, ctx);
  }
};

}  // namespace emel::text::renderer::guard
