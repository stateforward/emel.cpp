#pragma once

#include "emel/token/batcher/actions.hpp"
#include "emel/error/error.hpp"
#include "emel/token/batcher/context.hpp"
#include "emel/token/batcher/errors.hpp"
#include "emel/token/batcher/events.hpp"

namespace emel::token::batcher::guard {

namespace detail_guard {

inline bool required_outputs_present(const event::batch & req) noexcept {
  static_cast<void>(req);
  return true;
}

inline bool token_counts_valid(const event::batch & req) noexcept {
  return req.n_tokens > 0 && req.n_tokens <= action::MAX_TOKENS;
}

inline bool capacities_valid(const event::batch & req) noexcept {
  const int32_t mask_words = emel::token::batcher::detail::effective_mask_words(req);
  const int32_t stride = emel::token::batcher::detail::positions_stride(req);
  if (stride < 0) {
    return false;
  }
  const int32_t positions_count = stride == 3 ? req.n_tokens * 3 : req.n_tokens;
  return req.seq_primary_ids_capacity >= req.n_tokens &&
         req.seq_masks_capacity >= req.n_tokens * mask_words &&
         req.positions_capacity >= positions_count &&
         req.output_mask_capacity >= req.n_tokens;
}

inline bool token_ids_in_vocab(const event::batch & req) noexcept {
  const int32_t * token_ids = emel::token::batcher::detail::token_ids_ptr(req);
  if (req.vocab_size < 0) {
    return false;
  }
  if (req.vocab_size == 0) {
    return true;
  }

  for (int32_t i = 0; i < req.n_tokens; ++i) {
    const int32_t token_id = token_ids[i];
    if (token_id < 0 || token_id >= req.vocab_size) {
      return false;
    }
  }
  return true;
}

inline bool seq_payload_valid(const event::batch & req) noexcept {
  const bool has_masks = emel::token::batcher::detail::has_seq_masks_input(req);
  const bool has_primary = emel::token::batcher::detail::has_seq_primary_input(req);

  if (has_masks) {
    if (req.seq_mask_words <= 0 || req.seq_mask_words > action::SEQ_WORDS) {
      return false;
    }
    if (!emel::token::batcher::detail::masks_have_non_empty_rows(req)) {
      return false;
    }
  }

  const int32_t mask_words = emel::token::batcher::detail::effective_mask_words(req);
  const int32_t seq_limit = mask_words * 64;

  if (has_primary &&
      !emel::token::batcher::detail::primary_ids_in_range(
          req.seq_primary_ids, req.n_tokens, seq_limit)) {
    return false;
  }

  if (has_masks && has_primary &&
      !emel::token::batcher::detail::primary_in_mask_when_both_inputs(req)) {
    return false;
  }

  return true;
}

}  // namespace detail_guard

struct valid_request {
  bool operator()(const event::batch_runtime & ev, const action::context &) const noexcept {
    return detail_guard::required_outputs_present(ev.request) &&
           detail_guard::token_counts_valid(ev.request) &&
           detail_guard::capacities_valid(ev.request) &&
           detail_guard::token_ids_in_vocab(ev.request) &&
           detail_guard::seq_payload_valid(ev.request);
  }
};

struct invalid_request {
  bool operator()(const event::batch_runtime & ev, const action::context & ctx) const noexcept {
    return !valid_request{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct phase_failed {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err != emel::error::cast(error::none);
  }
};

struct seq_mode_masks {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::has_seq_masks_input(ev.request);
  }
};

struct seq_mode_primary_ids {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !emel::token::batcher::detail::has_seq_masks_input(ev.request) &&
           emel::token::batcher::detail::has_seq_primary_input(ev.request);
  }
};

struct seq_mode_default {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !emel::token::batcher::detail::has_seq_masks_input(ev.request) &&
           !emel::token::batcher::detail::has_seq_primary_input(ev.request);
  }
};

struct seq_mode_invalid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !seq_mode_masks{}(ev) &&
           !seq_mode_primary_ids{}(ev) &&
           !seq_mode_default{}(ev);
  }
};

struct positions_mode_stride_three {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) == 3;
  }
};

struct positions_mode_stride_one {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) == 1;
  }
};

struct positions_mode_generate_seeded {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) == 0 &&
           ev.request.resolve_position_seed != nullptr;
  }
};

struct positions_mode_generate_unseeded {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) == 0 &&
           ev.request.resolve_position_seed == nullptr;
  }
};

struct seeded_probe_ok {
  bool operator()(const event::batch_runtime &, const action::context & ctx) const noexcept {
    return ctx.seeded_probe_status == action::position_probe_status::ok;
  }
};

struct seeded_probe_backend_error {
  bool operator()(const event::batch_runtime &, const action::context & ctx) const noexcept {
    return ctx.seeded_probe_status == action::position_probe_status::backend_error;
  }
};

struct seeded_probe_invalid {
  bool operator()(const event::batch_runtime &, const action::context & ctx) const noexcept {
    return ctx.seeded_probe_status == action::position_probe_status::invalid;
  }
};

struct unseeded_probe_ok {
  bool operator()(const event::batch_runtime &, const action::context & ctx) const noexcept {
    return ctx.unseeded_probe_valid;
  }
};

struct unseeded_probe_invalid {
  bool operator()(const event::batch_runtime &, const action::context & ctx) const noexcept {
    return !ctx.unseeded_probe_valid;
  }
};

struct positions_mode_invalid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !positions_mode_stride_three{}(ev) &&
           !positions_mode_stride_one{}(ev) &&
           !positions_mode_generate_seeded{}(ev) &&
           !positions_mode_generate_unseeded{}(ev);
  }
};

struct output_mode_all {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.output_all;
  }
};

struct output_mode_copy {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !ev.request.output_all &&
           emel::token::batcher::detail::has_output_mask_input(ev.request);
  }
};

struct output_mode_last {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !ev.request.output_all &&
           !emel::token::batcher::detail::has_output_mask_input(ev.request);
  }
};

struct output_mode_invalid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !output_mode_all{}(ev) &&
           !output_mode_copy{}(ev) &&
           !output_mode_last{}(ev);
  }
};

struct single_output_check_required {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.enforce_single_output_per_seq;
  }
};

struct single_output_check_skipped {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !ev.request.enforce_single_output_per_seq;
  }
};

struct continuity_check_required {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) <= 1;
  }
};

struct continuity_check_skipped {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) > 1;
  }
};

struct single_output_check_passed {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return single_output_check_required{}(ev) &&
           emel::token::batcher::detail::single_output_per_seq_ok(ev);
  }
};

struct single_output_check_failed {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return single_output_check_required{}(ev) &&
           !emel::token::batcher::detail::single_output_per_seq_ok(ev);
  }
};

struct continuity_check_passed {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return continuity_check_required{}(ev) &&
           emel::token::batcher::detail::continuity_ok(ev);
  }
};

struct continuity_check_failed {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return continuity_check_required{}(ev) &&
           !emel::token::batcher::detail::continuity_ok(ev);
  }
};

struct seq_mask_words_out_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.seq_mask_words_out != nullptr;
  }
};

struct seq_mask_words_out_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !seq_mask_words_out_present{}(ev);
  }
};

struct positions_count_out_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.positions_count_out != nullptr;
  }
};

struct positions_count_out_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !positions_count_out_present{}(ev);
  }
};

struct outputs_total_out_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.outputs_total_out != nullptr;
  }
};

struct outputs_total_out_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !outputs_total_out_present{}(ev);
  }
};

struct done_callback_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct done_callback_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !done_callback_present{}(ev);
  }
};

struct error_callback_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

}  // namespace emel::token::batcher::guard
