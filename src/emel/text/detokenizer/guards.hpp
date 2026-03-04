#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/text/detokenizer/actions.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/detokenizer/events.hpp"

namespace emel::text::detokenizer::guard {

namespace detail {

inline size_t pending_head_sequence_length(const event::detokenize & ev) noexcept {
  return action::detail::utf8_sequence_length(ev.pending_bytes[0]);
}

inline bool pending_head_continuations_valid(const event::detokenize & ev,
                                             const size_t needed) noexcept {
  bool continuation_ok = true;
  for (size_t idx = 1; idx < needed; ++idx) {
    continuation_ok = continuation_ok &&
                      action::detail::is_utf8_continuation(ev.pending_bytes[idx]);
  }
  return continuation_ok;
}

}  // namespace detail

struct valid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    (void)ev;
    return true;
  }
};

struct invalid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    return !valid_bind{}(ev);
  }
};

struct valid_detokenize {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return ctx.is_bound && ctx.vocab != nullptr && ev.pending_bytes != nullptr &&
           ev.pending_capacity == action::detail::k_utf8_max_sequence_length &&
           ev.pending_length <= ev.pending_capacity &&
           (ev.output != nullptr || ev.output_capacity == 0);
  }
};

struct invalid_detokenize {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return !valid_detokenize{}(ev, ctx);
  }
};

struct detokenize_token_in_vocab {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return valid_detokenize{}(ev, ctx) && ev.token_id >= 0 &&
           static_cast<uint32_t>(ev.token_id) < ctx.vocab->n_tokens;
  }
};

struct detokenize_token_out_of_vocab {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return !detokenize_token_in_vocab{}(ev, ctx);
  }
};

struct detokenize_skip_special_piece {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return detokenize_token_in_vocab{}(ev, ctx) && !ev.emit_special &&
           action::detail::is_special_token_type(
               ctx.vocab->entries[static_cast<uint32_t>(ev.token_id)].type);
  }
};

struct detokenize_byte_piece {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    uint8_t byte_value = 0;
    const bool decode_piece = detokenize_token_in_vocab{}(ev, ctx) &&
                              !detokenize_skip_special_piece{}(ev, ctx);
    return decode_piece && action::detail::parse_plamo2_byte_token(
                               action::detail::token_piece(ev, ctx),
                               byte_value);
  }
};

struct detokenize_text_piece {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    const bool decode_piece = detokenize_token_in_vocab{}(ev, ctx) &&
                              !detokenize_skip_special_piece{}(ev, ctx);
    return decode_piece && !detokenize_byte_piece{}(ev, ctx);
  }
};

struct detokenize_pending_has_capacity_for_byte {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return detokenize_byte_piece{}(ev, ctx) &&
           ev.pending_length_out < ev.pending_capacity;
  }
};

struct detokenize_pending_no_capacity_for_byte {
  bool operator()(const event::detokenize & ev,
                  const action::context & ctx) const noexcept {
    return detokenize_byte_piece{}(ev, ctx) &&
           !detokenize_pending_has_capacity_for_byte{}(ev, ctx);
  }
};

struct bind_phase_ok {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.error_out == error_code(error::none);
  }
};

struct bind_phase_failed {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.error_out != error_code(error::none);
  }
};

struct detokenize_phase_ok {
  bool operator()(const event::detokenize & ev) const noexcept {
    return ev.error_out == error_code(error::none);
  }
};

struct detokenize_phase_failed {
  bool operator()(const event::detokenize & ev) const noexcept {
    return ev.error_out != error_code(error::none);
  }
};

struct detokenize_pending_empty {
  bool operator()(const event::detokenize & ev) const noexcept {
    return detokenize_phase_ok{}(ev) && ev.pending_length_out == 0;
  }
};

struct detokenize_pending_not_empty {
  bool operator()(const event::detokenize & ev) const noexcept {
    return detokenize_phase_ok{}(ev) && ev.pending_length_out != 0;
  }
};

struct detokenize_pending_head_complete {
  bool operator()(const event::detokenize & ev) const noexcept {
    const size_t needed = detail::pending_head_sequence_length(ev);
    const bool lead_ok = needed != 0;
    const bool sequence_ready =
        detokenize_pending_not_empty{}(ev) && lead_ok && ev.pending_length_out >= needed;
    return sequence_ready && detail::pending_head_continuations_valid(ev, needed);
  }
};

struct detokenize_pending_head_incomplete {
  bool operator()(const event::detokenize & ev) const noexcept {
    const size_t needed = detail::pending_head_sequence_length(ev);
    return detokenize_pending_not_empty{}(ev) && needed != 0 && ev.pending_length_out < needed;
  }
};

struct detokenize_pending_head_invalid {
  bool operator()(const event::detokenize & ev) const noexcept {
    const size_t needed = detail::pending_head_sequence_length(ev);
    const bool pending_not_empty = detokenize_pending_not_empty{}(ev);
    const bool lead_invalid = pending_not_empty && needed == 0;
    const bool sequence_ready = pending_not_empty && needed != 0 && ev.pending_length_out >= needed;
    const bool continuation_invalid =
        sequence_ready && !detail::pending_head_continuations_valid(ev, needed);
    return lead_invalid || continuation_invalid;
  }
};

struct has_bind_done_callback {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.dispatch_done != nullptr && ev.owner_sm != nullptr;
  }
};

struct no_bind_done_callback {
  bool operator()(const event::bind & ev) const noexcept {
    return !has_bind_done_callback{}(ev);
  }
};

struct has_bind_error_callback {
  bool operator()(const event::bind & ev) const noexcept {
    return ev.dispatch_error != nullptr && ev.owner_sm != nullptr;
  }
};

struct no_bind_error_callback {
  bool operator()(const event::bind & ev) const noexcept {
    return !has_bind_error_callback{}(ev);
  }
};

struct has_detokenize_done_callback {
  bool operator()(const event::detokenize & ev) const noexcept {
    return ev.dispatch_done != nullptr && ev.owner_sm != nullptr;
  }
};

struct no_detokenize_done_callback {
  bool operator()(const event::detokenize & ev) const noexcept {
    return !has_detokenize_done_callback{}(ev);
  }
};

struct has_detokenize_error_callback {
  bool operator()(const event::detokenize & ev) const noexcept {
    return ev.dispatch_error != nullptr && ev.owner_sm != nullptr;
  }
};

struct no_detokenize_error_callback {
  bool operator()(const event::detokenize & ev) const noexcept {
    return !has_detokenize_error_callback{}(ev);
  }
};

}  // namespace emel::text::detokenizer::guard
