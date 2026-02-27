#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "emel/emel.h"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/renderer/context.hpp"
#include "emel/text/renderer/events.hpp"

namespace emel::text::renderer::action {

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

inline void clear_request(context & ctx) noexcept {
  ctx.token_id = -1;
  ctx.sequence_id = 0;
  ctx.emit_special = false;
  ctx.output = nullptr;
  ctx.output_capacity = 0;
  ctx.output_length = 0;
  ctx.status = sequence_status::running;
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

inline bool resolve_sequence(context & ctx,
                             size_t & index_out) noexcept {
  if (ctx.sequence_id < 0) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    return false;
  }
  const size_t index = static_cast<size_t>(ctx.sequence_id);
  if (index >= ctx.sequences.size()) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    return false;
  }
  index_out = index;
  return true;
}

inline bool is_leading_space(const char value) noexcept {
  return value == ' ' || value == '\t' || value == '\n' || value == '\r';
}

inline char concat_char(const sequence_state & sequence,
                        const char * new_bytes,
                        const size_t index) noexcept {
  if (index < sequence.holdback_length) {
    return sequence.holdback[index];
  }
  return new_bytes[index - sequence.holdback_length];
}

inline bool copy_stop_sequences(const event::bind & ev,
                                context & ctx) noexcept {
  ctx.stop_sequence_count = 0;
  ctx.stop_storage_used = 0;
  ctx.stop_max_length = 0;

  if (ev.stop_sequence_count == 0) {
    return true;
  }
  if (ev.stop_sequences == nullptr ||
      ev.stop_sequence_count > ctx.stop_sequences.size()) {
    return false;
  }

  for (size_t index = 0; index < ev.stop_sequence_count; ++index) {
    const std::string_view stop = ev.stop_sequences[index];
    if (stop.empty()) {
      continue;
    }
    if (stop.size() > k_max_stop_length ||
        ctx.stop_storage_used + stop.size() > ctx.stop_storage.size() ||
        ctx.stop_sequence_count >= ctx.stop_sequences.size()) {
      return false;
    }

    const uint16_t offset = static_cast<uint16_t>(ctx.stop_storage_used);
    std::memcpy(ctx.stop_storage.data() + ctx.stop_storage_used,
                stop.data(),
                stop.size());
    ctx.stop_storage_used += stop.size();

    stop_sequence_entry entry = {};
    entry.offset = offset;
    entry.length = static_cast<uint16_t>(stop.size());
    ctx.stop_sequences[ctx.stop_sequence_count] = entry;
    ctx.stop_sequence_count += 1;
    ctx.stop_max_length = std::max(ctx.stop_max_length, stop.size());
  }

  return true;
}

inline bool compose_output(const sequence_state & sequence,
                           context & ctx,
                           const size_t emit_from_holdback,
                           const size_t emit_from_new,
                           const size_t new_length) noexcept {
  const size_t emit_total = emit_from_holdback + emit_from_new;
  if (emit_total > ctx.output_capacity || emit_from_new > new_length) {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    return false;
  }

  if (emit_from_holdback > 0) {
    if (emit_from_new > 0) {
      std::memmove(ctx.output + emit_from_holdback,
                   ctx.output,
                   emit_from_new);
    }
    std::memcpy(ctx.output, sequence.holdback.data(), emit_from_holdback);
  }

  ctx.output_length = emit_total;
  return true;
}

inline bool apply_stop_matching(sequence_state & sequence,
                                context & ctx,
                                const size_t new_length) noexcept {
  const size_t total = sequence.holdback_length + new_length;

  size_t matched_start = total;
  size_t matched_length = 0;

  if (ctx.stop_sequence_count > 0) {
    for (size_t stop_index = 0; stop_index < ctx.stop_sequence_count;
         ++stop_index) {
      const stop_sequence_entry stop = ctx.stop_sequences[stop_index];
      const size_t stop_length = static_cast<size_t>(stop.length);
      if (stop_length == 0 || stop_length > total) {
        continue;
      }

      for (size_t cursor = 0; cursor + stop_length <= total; ++cursor) {
        bool matched = true;
        for (size_t offset = 0; offset < stop_length; ++offset) {
          const char lhs = concat_char(sequence, ctx.output, cursor + offset);
          const char rhs =
              ctx.stop_storage[static_cast<size_t>(stop.offset) + offset];
          if (lhs != rhs) {
            matched = false;
            break;
          }
        }

        if (matched && cursor < matched_start) {
          matched_start = cursor;
          matched_length = stop_length;
        }
      }
    }
  }

  if (matched_length > 0) {
    const size_t emit_before_stop = matched_start;
    const size_t emit_from_holdback =
        std::min(emit_before_stop, sequence.holdback_length);
    const size_t emit_from_new = emit_before_stop - emit_from_holdback;

    if (!compose_output(sequence, ctx, emit_from_holdback, emit_from_new,
                        new_length)) {
      return false;
    }

    sequence.holdback_length = 0;
    sequence.stop_matched = true;
    ctx.status = sequence_status::stop_sequence_matched;
    return true;
  }

  const size_t holdback_target =
      ctx.stop_max_length > 1
          ? std::min(total, static_cast<size_t>(ctx.stop_max_length - 1))
          : 0;
  const size_t emit_total = total - holdback_target;
  const size_t emit_from_holdback =
      std::min(emit_total, sequence.holdback_length);
  const size_t emit_from_new = emit_total - emit_from_holdback;

  std::array<char, k_max_holdback_bytes> next_holdback = {};
  for (size_t idx = 0; idx < holdback_target; ++idx) {
    next_holdback[idx] =
        concat_char(sequence, ctx.output, total - holdback_target + idx);
  }

  if (!compose_output(sequence, ctx, emit_from_holdback, emit_from_new,
                      new_length)) {
    return false;
  }

  sequence.holdback_length = holdback_target;
  if (holdback_target > 0) {
    std::memcpy(sequence.holdback.data(), next_holdback.data(), holdback_target);
  }
  ctx.status = sequence_status::running;
  return true;
}

struct begin_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.vocab = ev.vocab;
    ctx.detokenizer_sm = ev.detokenizer_sm;
    ctx.dispatch_detokenizer_bind = ev.dispatch_detokenizer_bind;
    ctx.dispatch_detokenizer_detokenize = ev.dispatch_detokenizer_detokenize;
    ctx.strip_leading_space_default = ev.strip_leading_space;
    ctx.is_bound = false;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    clear_request(ctx);

    if (!copy_stop_sequences(ev, ctx)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    reset_sequences(ctx);
  }
};

struct reject_bind {
  void operator()(const event::bind &, context & ctx) const noexcept {
    ctx.is_bound = false;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct bind_detokenizer {
  void operator()(context & ctx) const noexcept {
    ctx.is_bound = false;
    if (ctx.phase_error != EMEL_OK) {
      if (ctx.last_error == EMEL_OK) {
        ctx.last_error = ctx.phase_error;
      }
      return;
    }
    ctx.phase_error = EMEL_OK;

    if (ctx.vocab == nullptr || ctx.detokenizer_sm == nullptr ||
        ctx.dispatch_detokenizer_bind == nullptr ||
        ctx.dispatch_detokenizer_detokenize == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    const int32_t detok_ok =
        emel::text::detokenizer::error_code(emel::text::detokenizer::error::none);
    const int32_t detok_backend =
        emel::text::detokenizer::error_code(emel::text::detokenizer::error::backend_error);
    int32_t err = detok_ok;

    emel::text::detokenizer::event::bind bind_ev = {};
    bind_ev.vocab = ctx.vocab;
    bind_ev.error_out = &err;

    const bool accepted =
        ctx.dispatch_detokenizer_bind(ctx.detokenizer_sm, bind_ev);
    if (!accepted && err == detok_ok) {
      set_error(ctx, detok_backend);
      return;
    }
    if (err != detok_ok) {
      set_error(ctx, err);
      return;
    }

    ctx.is_bound = true;
    ctx.last_error = EMEL_OK;
  }
};

struct begin_render {
  void operator()(const event::render & ev, context & ctx) const noexcept {
    if (ev.output_length_out != nullptr) {
      *ev.output_length_out = 0;
    }
    if (ev.status_out != nullptr) {
      *ev.status_out = sequence_status::running;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }

    ctx.token_id = ev.token_id;
    ctx.sequence_id = ev.sequence_id;
    ctx.emit_special = ev.emit_special;
    ctx.output = ev.output;
    ctx.output_capacity = ev.output_capacity;
    ctx.output_length = 0;
    ctx.status = sequence_status::running;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct reject_render {
  void operator()(const event::render &, context & ctx) const noexcept {
    ctx.output_length = 0;
    ctx.status = sequence_status::running;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct run_render {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.output_length = 0;
    ctx.status = sequence_status::running;

    if (!ctx.is_bound || ctx.vocab == nullptr ||
        (ctx.output == nullptr && ctx.output_capacity > 0) ||
        ctx.detokenizer_sm == nullptr ||
        ctx.dispatch_detokenizer_detokenize == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    size_t sequence_index = 0;
    if (!resolve_sequence(ctx, sequence_index)) {
      return;
    }

    sequence_state & sequence = ctx.sequences[sequence_index];
    if (sequence.stop_matched) {
      ctx.output_length = 0;
      ctx.status = sequence_status::stop_sequence_matched;
      ctx.last_error = EMEL_OK;
      return;
    }

    size_t pending_length = sequence.pending_length;
    size_t detok_output_length = 0;

    const int32_t detok_ok =
        emel::text::detokenizer::error_code(emel::text::detokenizer::error::none);
    const int32_t detok_backend =
        emel::text::detokenizer::error_code(emel::text::detokenizer::error::backend_error);
    int32_t err = detok_ok;

    emel::text::detokenizer::event::detokenize detok_ev = {};
    detok_ev.token_id = ctx.token_id;
    detok_ev.emit_special = ctx.emit_special;
    detok_ev.pending_bytes = sequence.pending_bytes.data();
    detok_ev.pending_length = sequence.pending_length;
    detok_ev.pending_capacity = sequence.pending_bytes.size();
    detok_ev.output = ctx.output;
    detok_ev.output_capacity = ctx.output_capacity;
    detok_ev.output_length_out = &detok_output_length;
    detok_ev.pending_length_out = &pending_length;
    detok_ev.error_out = &err;

    const bool accepted =
        ctx.dispatch_detokenizer_detokenize(ctx.detokenizer_sm, detok_ev);
    if (!accepted && err == detok_ok) {
      set_error(ctx, detok_backend);
      return;
    }
    if (err != detok_ok) {
      set_error(ctx, err);
      return;
    }
    if (detok_output_length > ctx.output_capacity ||
        pending_length > detok_ev.pending_capacity) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    sequence.pending_length = pending_length;

    size_t produced = detok_output_length;
    if (sequence.strip_leading_space && produced > 0) {
      size_t strip_count = 0;
      while (strip_count < produced && is_leading_space(ctx.output[strip_count])) {
        strip_count += 1;
      }
      if (strip_count > 0) {
        std::memmove(ctx.output,
                     ctx.output + strip_count,
                     produced - strip_count);
        produced -= strip_count;
      }
      if (produced > 0) {
        sequence.strip_leading_space = false;
      }
    }

    if (!apply_stop_matching(sequence, ctx, produced)) {
      return;
    }

    ctx.last_error = EMEL_OK;
  }
};

struct begin_flush {
  void operator()(const event::flush & ev, context & ctx) const noexcept {
    if (ev.output_length_out != nullptr) {
      *ev.output_length_out = 0;
    }
    if (ev.status_out != nullptr) {
      *ev.status_out = sequence_status::running;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }

    ctx.token_id = -1;
    ctx.sequence_id = ev.sequence_id;
    ctx.emit_special = false;
    ctx.output = ev.output;
    ctx.output_capacity = ev.output_capacity;
    ctx.output_length = 0;
    ctx.status = sequence_status::running;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct reject_flush {
  void operator()(const event::flush &, context & ctx) const noexcept {
    ctx.output_length = 0;
    ctx.status = sequence_status::running;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct run_flush {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.output_length = 0;
    ctx.status = sequence_status::running;

    if (!ctx.is_bound || ctx.vocab == nullptr ||
        (ctx.output == nullptr && ctx.output_capacity > 0)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    size_t sequence_index = 0;
    if (!resolve_sequence(ctx, sequence_index)) {
      return;
    }

    sequence_state & sequence = ctx.sequences[sequence_index];
    const size_t required = sequence.pending_length + sequence.holdback_length;
    if (required > ctx.output_capacity) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    if (sequence.pending_length > 0) {
      std::memcpy(ctx.output,
                  sequence.pending_bytes.data(),
                  sequence.pending_length);
      ctx.output_length += sequence.pending_length;
      sequence.pending_length = 0;
    }

    if (sequence.holdback_length > 0) {
      std::memcpy(ctx.output + ctx.output_length,
                  sequence.holdback.data(),
                  sequence.holdback_length);
      ctx.output_length += sequence.holdback_length;
      sequence.holdback_length = 0;
    }

    if (ctx.output_length > 0) {
      sequence.strip_leading_space = false;
    }

    ctx.status =
        sequence.stop_matched ? sequence_status::stop_sequence_matched
                              : sequence_status::running;
    ctx.last_error = EMEL_OK;
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error =
        ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context & ctx) const noexcept {
    ctx.output_length = 0;
    ctx.status = sequence_status::running;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

inline constexpr begin_bind begin_bind{};
inline constexpr reject_bind reject_bind{};
inline constexpr bind_detokenizer bind_detokenizer{};
inline constexpr begin_render begin_render{};
inline constexpr reject_render reject_render{};
inline constexpr run_render run_render{};
inline constexpr begin_flush begin_flush{};
inline constexpr reject_flush reject_flush{};
inline constexpr run_flush run_flush{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::renderer::action
