#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/text/tokenizer/preprocessor/context.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"
#include "emel/text/tokenizer/preprocessor/events.hpp"

namespace emel::text::tokenizer::preprocessor::action {

namespace pdetail = emel::text::tokenizer::preprocessor::detail;

namespace detail {

template <class runtime_event_type>
inline void clear_runtime(const runtime_event_type & runtime_ev) noexcept {
  auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
  ev.request.fragment_count_out = 0;
  bool preprocessed_sink = false;
  pdetail::write_optional(ev.request.preprocessed_out, preprocessed_sink, false);
  ev.request.error_out = preprocessor::error_code(preprocessor::error::none);
  ev.ctx.fragment_count = 0;
  ev.ctx.preprocessed = false;
  ev.ctx.phase_error = preprocessor::error::none;
  ev.ctx.err = preprocessor::error::none;
  ev.ctx.result = false;
}

template <class runtime_event_type>
inline void set_phase_error(const runtime_event_type & runtime_ev,
                            const preprocessor::error err) noexcept {
  auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
  ev.ctx.fragment_count = 0;
  ev.ctx.preprocessed = false;
  ev.ctx.phase_error = err;
  ev.ctx.err = err;
  ev.ctx.result = false;
}

template <class runtime_event_type>
inline void set_phase_result(const runtime_event_type & runtime_ev, const bool ok,
                             const size_t fragment_count,
                             const bool preprocessed) noexcept {
  auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
  const size_t idx = static_cast<size_t>(ok);
  const std::array<preprocessor::error, 2> errors = {
      preprocessor::error::invalid_request, preprocessor::error::none};
  const std::array<size_t, 2> counts = {0, fragment_count};
  const std::array<bool, 2> preprocessed_states = {false, preprocessed};
  ev.ctx.fragment_count = counts[idx];
  ev.ctx.preprocessed = preprocessed_states[idx];
  ev.ctx.phase_error = errors[idx];
  ev.ctx.err = errors[idx];
  ev.ctx.result = false;
}

}  // namespace detail

inline void clear_request(context &) noexcept {}

struct begin_preprocess {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    detail::clear_runtime(runtime_ev);
    ctx.bpe_scratch.reset();
  }
};

struct reject_invalid {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    detail::set_phase_error(runtime_ev, preprocessor::error::invalid_request);
  }
};

struct build_specials {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev, context & ctx) const {
    const auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    const bool ok = pdetail::build_special_tokens(ctx.special_cache, ev.request.vocab);
    detail::set_phase_result(runtime_ev, ok, 0, false);
  }
};

struct mark_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    ev.ctx.phase_error = preprocessor::error::none;
    ev.ctx.err = preprocessor::error::none;
    ev.ctx.result = true;
  }
};

struct ensure_last_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = preprocessor::resolve_failure_error(ev.ctx.phase_error);
    ev.ctx.result = false;
  }
};

struct on_unexpected {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = pdetail::unwrap_runtime_event(runtime_ev);
    if constexpr (requires { ev.ctx.fragment_count; }) {
      ev.ctx.fragment_count = 0;
    }
    if constexpr (requires { ev.ctx.preprocessed; }) {
      ev.ctx.preprocessed = false;
    }
    if constexpr (requires { ev.ctx.phase_error; }) {
      ev.ctx.phase_error = preprocessor::error::invalid_request;
    }
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = preprocessor::error::invalid_request;
    }
    if constexpr (requires { ev.ctx.result; }) {
      ev.ctx.result = false;
    }
  }
};

inline constexpr begin_preprocess begin_preprocess{};
inline constexpr reject_invalid reject_invalid{};
inline constexpr build_specials build_specials{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::tokenizer::preprocessor::action
