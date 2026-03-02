#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "emel/error/error.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/renderer/context.hpp"
#include "emel/text/renderer/errors.hpp"
#include "emel/text/renderer/events.hpp"

namespace emel::text::renderer::action {

namespace detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

}  // namespace detail

inline constexpr emel::error::type k_error_none = emel::error::cast(error::none);

inline constexpr int32_t to_detokenizer_error_code(
    const emel::text::detokenizer::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

inline constexpr int32_t k_detokenizer_ok =
    to_detokenizer_error_code(emel::text::detokenizer::error::none);
inline constexpr int32_t k_detokenizer_backend_error =
    to_detokenizer_error_code(emel::text::detokenizer::error::backend_error);

inline constexpr int32_t to_error_out(const emel::error::type err) noexcept {
  return static_cast<int32_t>(err);
}

struct detokenizer_error_map_entry {
  int32_t detokenizer_code = k_detokenizer_ok;
  emel::error::type renderer_code = emel::error::cast(error::none);
};

inline emel::error::type from_detokenizer_error(const int32_t err) noexcept {
  constexpr std::array<detokenizer_error_map_entry, 5> k_error_map = {{
      {to_detokenizer_error_code(emel::text::detokenizer::error::none),
       emel::error::cast(error::none)},
      {to_detokenizer_error_code(emel::text::detokenizer::error::invalid_request),
       emel::error::cast(error::invalid_request)},
      {to_detokenizer_error_code(emel::text::detokenizer::error::model_invalid),
       emel::error::cast(error::model_invalid)},
      {to_detokenizer_error_code(emel::text::detokenizer::error::backend_error),
       emel::error::cast(error::backend_error)},
      {to_detokenizer_error_code(emel::text::detokenizer::error::internal_error),
       emel::error::cast(error::internal_error)},
  }};

  emel::error::type mapped = emel::error::cast(error::untracked);
  for (const auto & entry : k_error_map) {
    const size_t matched = static_cast<size_t>(entry.detokenizer_code == err);
    const emel::error::type candidates[2] = {mapped, entry.renderer_code};
    mapped = candidates[matched];
  }
  return mapped;
}

template <class runtime_ctx_type>
inline void set_error(runtime_ctx_type & runtime_ctx,
                      const emel::error::type err) noexcept {
  runtime_ctx.err = err;
}

template <class runtime_ctx_type>
inline void set_error(runtime_ctx_type & runtime_ctx,
                      const error err) noexcept {
  set_error(runtime_ctx, emel::error::cast(err));
}

template <class runtime_ctx_type>
inline void reset_outcome(runtime_ctx_type & runtime_ctx) noexcept {
  runtime_ctx.err = k_error_none;
  if constexpr (requires { runtime_ctx.output_length; }) {
    runtime_ctx.output_length = 0;
  }
  if constexpr (requires { runtime_ctx.status; }) {
    runtime_ctx.status = sequence_status::running;
  }
  if constexpr (requires { runtime_ctx.sequence_index; }) {
    runtime_ctx.sequence_index = 0;
  }
  if constexpr (requires { runtime_ctx.detokenizer_err; }) {
    runtime_ctx.detokenizer_err = k_detokenizer_ok;
  }
  if constexpr (requires { runtime_ctx.detokenizer_output_length; }) {
    runtime_ctx.detokenizer_output_length = 0;
  }
  if constexpr (requires { runtime_ctx.detokenizer_pending_length; }) {
    runtime_ctx.detokenizer_pending_length = 0;
  }
  if constexpr (requires { runtime_ctx.produced_length; }) {
    runtime_ctx.produced_length = 0;
  }
}

inline void reset_sequence_state(sequence_state & state,
                                 const bool strip_leading_space) noexcept {
  state.pending_length = 0;
  state.holdback_length = 0;
  state.strip_leading_space = strip_leading_space;
  state.stop_matched = false;
}

inline void reset_sequences(context & ctx) noexcept {
  for (auto & state : ctx.sequences) {
    reset_sequence_state(state, ctx.strip_leading_space_default);
  }
}

inline bool is_leading_space(const char value) noexcept {
  return value == ' ' || value == '\t' || value == '\n' || value == '\r';
}

inline char concat_char(const sequence_state & sequence,
                        const char * new_bytes,
                        const size_t index) noexcept {
  const size_t from_new = static_cast<size_t>(index >= sequence.holdback_length);
  const char * sources[2] = {sequence.holdback.data(), new_bytes};
  const size_t adjusted_indices[2] = {
      index,
      index - (sequence.holdback_length * from_new)};
  return sources[from_new][adjusted_indices[from_new]];
}

inline bool copy_stop_sequences(const event::initialize & ev,
                                context & ctx) noexcept {
  ctx.stop_sequence_count = ev.stop_sequence_count;
  ctx.stop_storage_used = 0;
  ctx.stop_max_length = 0;

  for (size_t index = 0; index < ev.stop_sequence_count; ++index) {
    const std::string_view stop = ev.stop_sequences[index];
    const uint16_t offset = static_cast<uint16_t>(ctx.stop_storage_used);
    std::memcpy(ctx.stop_storage.data() + ctx.stop_storage_used,
                stop.data(),
                stop.size());
    ctx.stop_storage_used += stop.size();

    stop_sequence_entry entry = {};
    entry.offset = offset;
    entry.length = static_cast<uint16_t>(stop.size());
    ctx.stop_sequences[index] = entry;
    ctx.stop_max_length = std::max(ctx.stop_max_length, stop.size());
  }

  return true;
}

template <class value_type>
inline void write_optional(value_type * destination,
                           value_type & sink,
                           const value_type value) noexcept {
  value_type * destinations[2] = {&sink, destination};
  value_type * const target =
      destinations[static_cast<size_t>(destination != nullptr)];
  *target = value;
}

template <class runtime_ctx_type>
inline bool compose_output(const sequence_state & sequence,
                           char * output,
                           const size_t output_capacity,
                           const size_t emit_from_holdback,
                           const size_t emit_from_new,
                           const size_t new_length,
                           size_t & output_length_out,
                           runtime_ctx_type & runtime_ctx) noexcept {
  const size_t emit_total = emit_from_holdback + emit_from_new;
  const size_t output_required = static_cast<size_t>(emit_total > 0);
  const size_t has_output = static_cast<size_t>(output != nullptr);
  const size_t output_ready = has_output | static_cast<size_t>(output_required == 0);
  const size_t total_within_bounds = static_cast<size_t>(emit_total <= output_capacity);
  const size_t new_within_bounds = static_cast<size_t>(emit_from_new <= new_length);
  const size_t valid = output_ready & total_within_bounds & new_within_bounds;

  const std::array<emel::error::type, 2> errors = {
      emel::error::cast(error::invalid_request),
      emel::error::cast(error::none)};
  set_error(runtime_ctx, errors[valid]);

  std::array<char, k_max_pending_bytes + k_max_holdback_bytes> output_sink = {};
  char * destinations[2] = {output_sink.data(), output};
  char * const target = destinations[has_output & valid];
  const size_t safe_emit_from_holdback = emit_from_holdback * valid;
  const size_t safe_emit_from_new = emit_from_new * valid;

  std::memmove(target + safe_emit_from_holdback,
               target,
               safe_emit_from_new);
  std::memcpy(target, sequence.holdback.data(), safe_emit_from_holdback);

  output_length_out = emit_total * valid;
  return valid != 0;
}

template <class runtime_ctx_type>
inline bool apply_stop_matching(sequence_state & sequence,
                                context & ctx,
                                char * output,
                                const size_t output_capacity,
                                const size_t new_length,
                                size_t & output_length_out,
                                sequence_status & status_out,
                                runtime_ctx_type & runtime_ctx) noexcept {
  const size_t total = sequence.holdback_length + new_length;

  size_t matched_start = total;
  size_t matched_length = 0;

  for (size_t stop_index = 0; stop_index < ctx.stop_sequence_count;
       ++stop_index) {
    const stop_sequence_entry stop = ctx.stop_sequences[stop_index];
    const size_t stop_length = static_cast<size_t>(stop.length);
    const size_t bounded_stop_length = std::min(stop_length, total + 1);
    const size_t stop_non_zero = static_cast<size_t>(stop_length != 0);
    const size_t cursor_limit = (total + 1 - bounded_stop_length) * stop_non_zero;

    for (size_t cursor = 0; cursor < cursor_limit; ++cursor) {
      bool matched = true;
      for (size_t offset = 0; offset < stop_length; ++offset) {
        const char lhs = concat_char(sequence, output, cursor + offset);
        const char rhs =
            ctx.stop_storage[static_cast<size_t>(stop.offset) + offset];
        matched = matched && (lhs == rhs);
      }

      const size_t should_update =
          static_cast<size_t>(matched) & static_cast<size_t>(cursor < matched_start);
      const size_t starts[2] = {matched_start, cursor};
      const size_t lengths[2] = {matched_length, stop_length};
      matched_start = starts[should_update];
      matched_length = lengths[should_update];
    }
  }

  const size_t matched = static_cast<size_t>(matched_length > 0);
  const size_t emit_before_stop_match = matched_start;
  const size_t emit_from_holdback_match =
      std::min(emit_before_stop_match, sequence.holdback_length);
  const size_t emit_from_new_match = emit_before_stop_match - emit_from_holdback_match;
  const size_t holdback_target_match = 0;

  const size_t has_holdback_window = static_cast<size_t>(ctx.stop_max_length > 1);
  const size_t holdback_limits[2] = {0, ctx.stop_max_length - 1};
  const size_t holdback_target_nomatch = std::min(total, holdback_limits[has_holdback_window]);
  const size_t emit_total_nomatch = total - holdback_target_nomatch;
  const size_t emit_from_holdback_nomatch =
      std::min(emit_total_nomatch, sequence.holdback_length);
  const size_t emit_from_new_nomatch = emit_total_nomatch - emit_from_holdback_nomatch;

  std::array<char, k_max_holdback_bytes> next_holdback = {};
  for (size_t idx = 0; idx < holdback_target_nomatch; ++idx) {
    next_holdback[idx] =
        concat_char(sequence, output, total - holdback_target_nomatch + idx);
  }

  const size_t emit_from_holdback_options[2] = {
      emit_from_holdback_nomatch,
      emit_from_holdback_match};
  const size_t emit_from_new_options[2] = {
      emit_from_new_nomatch,
      emit_from_new_match};
  const size_t holdback_targets[2] = {
      holdback_target_nomatch,
      holdback_target_match};
  const bool stop_matched_values[2] = {false, true};
  const std::array<sequence_status, 2> statuses = {
      sequence_status::running,
      sequence_status::stop_sequence_matched};

  const size_t selected_emit_from_holdback = emit_from_holdback_options[matched];
  const size_t selected_emit_from_new = emit_from_new_options[matched];
  const size_t selected_holdback_target = holdback_targets[matched];
  const bool selected_stop_matched = stop_matched_values[matched];
  const sequence_status selected_status = statuses[matched];

  const size_t compose_ok = static_cast<size_t>(
      compose_output(sequence,
                     output,
                     output_capacity,
                     selected_emit_from_holdback,
                     selected_emit_from_new,
                     new_length,
                     output_length_out,
                     runtime_ctx));

  const size_t holdback_lengths_after_compose[2] = {
      sequence.holdback_length,
      selected_holdback_target};
  sequence.holdback_length = holdback_lengths_after_compose[compose_ok];
  std::memcpy(sequence.holdback.data(),
              next_holdback.data(),
              selected_holdback_target * compose_ok);

  const bool stop_matched_after_compose[2] = {
      sequence.stop_matched,
      selected_stop_matched};
  sequence.stop_matched = stop_matched_after_compose[compose_ok];

  const std::array<sequence_status, 2> status_after_compose = {
      status_out,
      selected_status};
  status_out = status_after_compose[compose_ok];

  return compose_ok != 0;
}

struct begin_initialize {
  void operator()(const event::initialize_runtime & ev, context &) const noexcept {
    reset_outcome(ev.ctx);
    int32_t error_sink = to_error_out(k_error_none);
    write_optional(ev.request.error_out, error_sink, to_error_out(k_error_none));
  }
};

struct reject_initialize {
  void operator()(const event::initialize_runtime & ev, context &) const noexcept {
    set_error(ev.ctx, error::invalid_request);
  }
};

struct dispatch_initialize_detokenizer {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    int32_t err = k_detokenizer_ok;
    const emel::text::detokenizer::event::bind bind_ev{
        runtime_ev.request.vocab,
        err};

    const bool accepted = ctx.detokenizer.process_event(bind_ev);
    if (!accepted && err == k_detokenizer_ok) {
      err = k_detokenizer_backend_error;
    }
    runtime_ev.ctx.detokenizer_err = err;
  }
};

struct set_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    set_error(runtime_ev.ctx, error::backend_error);
  }
};

struct set_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    set_error(runtime_ev.ctx, error::invalid_request);
  }
};

struct set_error_from_detokenizer {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    set_error(runtime_ev.ctx, from_detokenizer_error(runtime_ev.ctx.detokenizer_err));
  }
};

struct commit_initialize_success {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    ctx.vocab = &runtime_ev.request.vocab;
    ctx.strip_leading_space_default = runtime_ev.request.strip_leading_space;
    copy_stop_sequences(runtime_ev.request, ctx);
    reset_sequences(ctx);
    set_error(runtime_ev.ctx, error::none);
  }
};

struct publish_initialize_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    int32_t error_sink = to_error_out(k_error_none);
    write_optional(ev.request.error_out,
                  error_sink,
                  to_error_out(ev.ctx.err));
    if (ev.request.owner_sm != nullptr &&
        ev.request.dispatch_done != nullptr) {
      ev.request.dispatch_done(ev.request.owner_sm,
                               events::initialize_done{&ev.request});
    }
  }
};

struct publish_initialize_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    int32_t error_sink = to_error_out(k_error_none);
    write_optional(ev.request.error_out,
                  error_sink,
                  to_error_out(ev.ctx.err));
    if (ev.request.owner_sm != nullptr &&
        ev.request.dispatch_error != nullptr) {
      ev.request.dispatch_error(ev.request.owner_sm,
                               events::initialize_error{&ev.request,
                                                       to_error_out(ev.ctx.err)});
    }
  }
};

struct begin_render {
  void operator()(const event::render_runtime & ev,
                  context &) const noexcept {
    reset_outcome(ev.ctx);
    int32_t error_sink = to_error_out(k_error_none);
    size_t output_length_sink = 0;
    sequence_status status_sink = sequence_status::running;
    write_optional(ev.request.output_length_out, output_length_sink, size_t{0});
    write_optional(ev.request.status_out, status_sink, sequence_status::running);
    write_optional(ev.request.error_out, error_sink, to_error_out(k_error_none));
    ev.ctx.sequence_index = static_cast<size_t>(ev.request.sequence_id);
  }
};

struct reject_render {
  void operator()(const event::render_runtime & ev,
                  context &) const noexcept {
    reset_outcome(ev.ctx);
    set_error(ev.ctx, error::invalid_request);
  }
};

struct render_sequence_already_stopped {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.output_length = 0;
    runtime_ev.ctx.status = sequence_status::stop_sequence_matched;
    runtime_ev.ctx.produced_length = 0;
    set_error(runtime_ev.ctx, error::none);
  }
};

struct dispatch_render_detokenizer {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    sequence_state & sequence = ctx.sequences[runtime_ev.ctx.sequence_index];

    int32_t err = k_detokenizer_ok;
    size_t detok_output_length = 0;
    size_t detok_pending_length = sequence.pending_length;

    const emel::text::detokenizer::event::detokenize detok_ev{
        runtime_ev.request.token_id,
        runtime_ev.request.emit_special,
        sequence.pending_bytes.data(),
        sequence.pending_length,
        sequence.pending_bytes.size(),
        runtime_ev.request.output,
        runtime_ev.request.output_capacity,
        detok_output_length,
        detok_pending_length,
        err};

    const bool accepted = ctx.detokenizer.process_event(detok_ev);
    if (!accepted && err == k_detokenizer_ok) {
      err = k_detokenizer_backend_error;
    }
    runtime_ev.ctx.detokenizer_err = err;
    runtime_ev.ctx.detokenizer_output_length = detok_output_length;
    runtime_ev.ctx.detokenizer_pending_length = detok_pending_length;
  }
};

struct commit_render_detokenizer_output {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    sequence_state & sequence = ctx.sequences[runtime_ev.ctx.sequence_index];
    sequence.pending_length = runtime_ev.ctx.detokenizer_pending_length;
    runtime_ev.ctx.produced_length = runtime_ev.ctx.detokenizer_output_length;
  }
};

struct strip_render_leading_space {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    size_t strip_count = 0;
    const size_t produced = runtime_ev.ctx.produced_length;
    while (strip_count < produced &&
           is_leading_space(runtime_ev.request.output[strip_count])) {
      strip_count += 1;
    }

    std::memmove(runtime_ev.request.output,
                 runtime_ev.request.output + strip_count,
                 produced - strip_count);
    runtime_ev.ctx.produced_length = produced - strip_count;
  }
};

struct update_render_strip_state {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    sequence_state & sequence = ctx.sequences[runtime_ev.ctx.sequence_index];
    const bool keep_strip = runtime_ev.ctx.produced_length == 0;
    sequence.strip_leading_space = sequence.strip_leading_space && keep_strip;
  }
};

struct apply_render_stop_matching {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    sequence_state & sequence = ctx.sequences[runtime_ev.ctx.sequence_index];
    apply_stop_matching(sequence,
                             ctx,
                             runtime_ev.request.output,
                             runtime_ev.request.output_capacity,
                             runtime_ev.ctx.produced_length,
                             runtime_ev.ctx.output_length,
                             runtime_ev.ctx.status,
                             runtime_ev.ctx);
  }
};

struct commit_render_output {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    commit_render_detokenizer_output{}(ev, ctx);
    update_render_strip_state{}(ev, ctx);
    apply_render_stop_matching{}(ev, ctx);
  }
};

struct commit_and_strip_render_output {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    commit_render_detokenizer_output{}(ev, ctx);
    strip_render_leading_space{}(ev, ctx);
    update_render_strip_state{}(ev, ctx);
    apply_render_stop_matching{}(ev, ctx);
  }
};

struct begin_flush {
  void operator()(const event::flush_runtime & ev,
                  context &) const noexcept {
    reset_outcome(ev.ctx);
    int32_t error_sink = to_error_out(k_error_none);
    size_t output_length_sink = 0;
    sequence_status status_sink = sequence_status::running;
    write_optional(ev.request.output_length_out, output_length_sink, size_t{0});
    write_optional(ev.request.status_out, status_sink, sequence_status::running);
    write_optional(ev.request.error_out, error_sink, to_error_out(k_error_none));
    ev.ctx.sequence_index = static_cast<size_t>(ev.request.sequence_id);
  }
};

struct reject_flush {
  void operator()(const event::flush_runtime & ev,
                  context &) const noexcept {
    reset_outcome(ev.ctx);
    set_error(ev.ctx, error::invalid_request);
  }
};

struct flush_copy_sequence_buffers {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context & ctx) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    sequence_state & sequence = ctx.sequences[runtime_ev.ctx.sequence_index];
    const size_t pending_length = sequence.pending_length;
    const size_t holdback_length = sequence.holdback_length;
    std::array<char, k_max_pending_bytes + k_max_holdback_bytes> output_sink = {};
    char * destinations[2] = {output_sink.data(), runtime_ev.request.output};
    char * const output =
        destinations[static_cast<size_t>(runtime_ev.request.output != nullptr)];

    std::memcpy(output,
                sequence.pending_bytes.data(),
                pending_length);
    std::memcpy(output + pending_length,
                sequence.holdback.data(),
                holdback_length);
    runtime_ev.ctx.output_length = pending_length + holdback_length;
    sequence.pending_length = 0;
    sequence.holdback_length = 0;

    const bool keep_strip = runtime_ev.ctx.output_length == 0;
    sequence.strip_leading_space = sequence.strip_leading_space && keep_strip;

    const std::array<sequence_status, 2> statuses = {
        sequence_status::running,
        sequence_status::stop_sequence_matched};
    runtime_ev.ctx.status = statuses[static_cast<size_t>(sequence.stop_matched)];
  }
};

struct publish_render_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    int32_t error_sink = to_error_out(k_error_none);
    size_t output_length_sink = 0;
    sequence_status status_sink = sequence_status::running;

    write_optional(ev.request.output_length_out,
                   output_length_sink,
                   ev.ctx.output_length);
    write_optional(ev.request.status_out, status_sink, ev.ctx.status);
    write_optional(ev.request.error_out,
                   error_sink,
                   to_error_out(ev.ctx.err));
    if (ev.request.owner_sm != nullptr && ev.request.dispatch_done != nullptr) {
      ev.request.dispatch_done(
          ev.request.owner_sm,
          events::rendering_done{&ev.request,
                                 ev.ctx.output_length,
                                 ev.ctx.status});
    }
  }
};

struct publish_render_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    int32_t error_sink = to_error_out(k_error_none);
    write_optional(ev.request.error_out,
                   error_sink,
                   to_error_out(ev.ctx.err));
    if (ev.request.owner_sm != nullptr &&
        ev.request.dispatch_error != nullptr) {
      ev.request.dispatch_error(
          ev.request.owner_sm,
          events::rendering_error{&ev.request, to_error_out(ev.ctx.err)});
    }
  }
};

struct publish_flush_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    int32_t error_sink = to_error_out(k_error_none);
    size_t output_length_sink = 0;
    sequence_status status_sink = sequence_status::running;

    write_optional(ev.request.output_length_out,
                   output_length_sink,
                   ev.ctx.output_length);
    write_optional(ev.request.status_out, status_sink, ev.ctx.status);
    write_optional(ev.request.error_out,
                   error_sink,
                   to_error_out(ev.ctx.err));
    if (ev.request.owner_sm != nullptr && ev.request.dispatch_done != nullptr) {
      ev.request.dispatch_done(
          ev.request.owner_sm,
          events::flush_done{&ev.request,
                             ev.ctx.output_length,
                             ev.ctx.status});
    }
  }
};

struct publish_flush_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    int32_t error_sink = to_error_out(k_error_none);
    write_optional(ev.request.error_out,
                   error_sink,
                   to_error_out(ev.ctx.err));
    if (ev.request.owner_sm != nullptr &&
        ev.request.dispatch_error != nullptr) {
      ev.request.dispatch_error(
          ev.request.owner_sm,
          events::flush_error{&ev.request, to_error_out(ev.ctx.err)});
    }
  }
};

struct mark_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    set_error(runtime_ev.ctx, error::none);
  }
};

struct ensure_last_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev,
                  context &) const noexcept {
    auto & runtime_ev = detail::unwrap_runtime_event(ev);
    const std::array<emel::error::type, 2> errors = {
        emel::error::cast(error::backend_error),
        runtime_ev.ctx.err};
    set_error(runtime_ev.ctx, errors[static_cast<size_t>(runtime_ev.ctx.err != k_error_none)]);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev,
                  context &) const noexcept {
    if constexpr (requires { detail::unwrap_runtime_event(ev).ctx; }) {
      auto & runtime_ev = detail::unwrap_runtime_event(ev);
      set_error(runtime_ev.ctx, error::invalid_request);
      if constexpr (requires { runtime_ev.ctx.output_length; }) {
        runtime_ev.ctx.output_length = 0;
      }
      if constexpr (requires { runtime_ev.ctx.status; }) {
        runtime_ev.ctx.status = sequence_status::running;
      }
    }
  }
};

inline constexpr begin_initialize begin_initialize{};
inline constexpr reject_initialize reject_initialize{};
inline constexpr dispatch_initialize_detokenizer dispatch_initialize_detokenizer{};
inline constexpr set_backend_error set_backend_error{};
inline constexpr set_invalid_request set_invalid_request{};
inline constexpr set_error_from_detokenizer set_error_from_detokenizer{};
inline constexpr commit_initialize_success commit_initialize_success{};
inline constexpr publish_initialize_done publish_initialize_done{};
inline constexpr publish_initialize_error publish_initialize_error{};
inline constexpr begin_render begin_render{};
inline constexpr reject_render reject_render{};
inline constexpr render_sequence_already_stopped render_sequence_already_stopped{};
inline constexpr dispatch_render_detokenizer dispatch_render_detokenizer{};
inline constexpr commit_render_detokenizer_output commit_render_detokenizer_output{};
inline constexpr commit_render_output commit_render_output{};
inline constexpr commit_and_strip_render_output commit_and_strip_render_output{};
inline constexpr strip_render_leading_space strip_render_leading_space{};
inline constexpr update_render_strip_state update_render_strip_state{};
inline constexpr apply_render_stop_matching apply_render_stop_matching{};
inline constexpr begin_flush begin_flush{};
inline constexpr reject_flush reject_flush{};
inline constexpr flush_copy_sequence_buffers flush_copy_sequence_buffers{};
inline constexpr publish_render_done publish_render_done{};
inline constexpr publish_render_error publish_render_error{};
inline constexpr publish_flush_done publish_flush_done{};
inline constexpr publish_flush_error publish_flush_error{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::renderer::action
