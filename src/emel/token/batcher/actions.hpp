#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include "emel/error/error.hpp"
#include "emel/token/batcher/context.hpp"
#include "emel/token/batcher/detail.hpp"
#include "emel/token/batcher/errors.hpp"
#include "emel/token/batcher/events.hpp"

namespace emel::token::batcher::action {

struct begin_batch {
  void operator()(const event::batch_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.outputs_total = 0;
    ev.ctx.normalized_seq_mask_words = detail::effective_mask_words(ev.request);
    ev.ctx.normalized_positions_count = ev.request.n_tokens;
    ctx.seeded_probe_status = position_probe_status::none;
    ctx.unseeded_probe_valid = false;
    ctx.seeded_next_pos.fill(0);
    detail::write_error(ev, ev.ctx.err);
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
    detail::write_error(runtime_ev, runtime_ev.ctx.err);
  }
};

struct mark_internal_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::internal_error);
    detail::write_error(runtime_ev, runtime_ev.ctx.err);
  }
};

struct mark_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::backend_error);
    detail::write_error(runtime_ev, runtime_ev.ctx.err);
  }
};

struct normalize_seq_from_masks {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    const int32_t mask_words = req.seq_mask_words;
    int32_t * seq_primary_ids_out = detail::seq_primary_ids_out_ptr(req);
    uint64_t * seq_masks_out = detail::seq_masks_out_ptr(req);

    ev.ctx.normalized_seq_mask_words = mask_words;

    for (int32_t i = 0; i < req.n_tokens; ++i) {
      const uint64_t * in_mask = req.seq_masks + static_cast<size_t>(i) * mask_words;
      uint64_t * out_mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
      std::copy_n(in_mask, mask_words, out_mask);
      seq_primary_ids_out[i] = detail::mask_primary_id(out_mask, mask_words);
    }
  }
};

struct normalize_seq_from_primary_ids {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    constexpr int32_t mask_words = 1;
    int32_t * seq_primary_ids_out = detail::seq_primary_ids_out_ptr(req);
    uint64_t * seq_masks_out = detail::seq_masks_out_ptr(req);

    ev.ctx.normalized_seq_mask_words = mask_words;

    for (int32_t i = 0; i < req.n_tokens; ++i) {
      const int32_t seq_id = req.seq_primary_ids[i];
      uint64_t * out_mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
      detail::clear_mask(out_mask, mask_words);
      detail::set_mask_bit(out_mask, mask_words, seq_id);
      seq_primary_ids_out[i] = seq_id;
    }
  }
};

struct normalize_seq_default {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    constexpr int32_t mask_words = 1;
    int32_t * seq_primary_ids_out = detail::seq_primary_ids_out_ptr(req);
    uint64_t * seq_masks_out = detail::seq_masks_out_ptr(req);

    ev.ctx.normalized_seq_mask_words = mask_words;

    for (int32_t i = 0; i < req.n_tokens; ++i) {
      uint64_t * out_mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
      detail::clear_mask(out_mask, mask_words);
      detail::set_mask_bit(out_mask, mask_words, 0);
      seq_primary_ids_out[i] = 0;
    }
  }
};

struct copy_positions_stride_three {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    int32_t * positions_out = detail::positions_out_ptr(req);
    const int32_t count = req.n_tokens * 3;
    std::copy_n(req.positions, count, positions_out);
    ev.ctx.normalized_positions_count = count;
  }
};

struct copy_positions_stride_one {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    int32_t * positions_out = detail::positions_out_ptr(req);
    const int32_t count = req.n_tokens;
    std::copy_n(req.positions, count, positions_out);
    ev.ctx.normalized_positions_count = count;
  }
};

struct probe_positions_seeded {
  void operator()(const event::batch_runtime & ev, context & ctx) const noexcept {
    const detail::probe_status status = detail::seeded_generation_probe(ev, ctx.seeded_next_pos);
    switch (status) {
      case detail::probe_status::ok:
        ctx.seeded_probe_status = position_probe_status::ok;
        break;
      case detail::probe_status::backend_error:
        ctx.seeded_probe_status = position_probe_status::backend_error;
        break;
      default:
        ctx.seeded_probe_status = position_probe_status::invalid;
        break;
    }
  }
};

struct probe_positions_unseeded {
  void operator()(const event::batch_runtime & ev, context & ctx) const noexcept {
    ctx.unseeded_probe_valid = detail::unseeded_generation_probe(ev);
  }
};

struct generate_positions_seeded {
  void operator()(const event::batch_runtime & ev, const context & ctx) const noexcept {
    const auto & req = ev.request;
    const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
    const int32_t * seq_primary_ids_out = detail::seq_primary_ids_out_ptr(req);
    uint64_t * seq_masks_out = detail::seq_masks_out_ptr(req);
    int32_t * positions_out = detail::positions_out_ptr(req);
    std::array<int32_t, MAX_SEQ> next_pos = ctx.seeded_next_pos;

    for (int32_t i = 0; i < req.n_tokens; ++i) {
      const int32_t primary = seq_primary_ids_out[i];
      const int32_t pos = next_pos[primary];
      positions_out[i] = pos;

      const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
      detail::for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
        next_pos[seq_id] = pos + 1;
        return true;
      });
    }

    ev.ctx.normalized_positions_count = req.n_tokens;
  }
};

struct generate_positions_unseeded {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
    const int32_t * seq_primary_ids_out = detail::seq_primary_ids_out_ptr(req);
    uint64_t * seq_masks_out = detail::seq_masks_out_ptr(req);
    int32_t * positions_out = detail::positions_out_ptr(req);
    std::array<int32_t, MAX_SEQ> next_pos = {};
    std::array<uint8_t, MAX_SEQ> seeded = {};

    for (int32_t i = 0; i < req.n_tokens; ++i) {
      const int32_t primary = seq_primary_ids_out[i];
      const int32_t pos = next_pos[primary];
      positions_out[i] = pos;

      const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
      detail::for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
        seeded[seq_id] = 1U;
        next_pos[seq_id] = pos + 1;
        return true;
      });
    }

    ev.ctx.normalized_positions_count = req.n_tokens;
  }
};

struct set_output_mask_all {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    int8_t * output_mask_out = detail::output_mask_out_ptr(req);
    std::fill_n(output_mask_out, req.n_tokens, static_cast<int8_t>(1));
  }
};

struct copy_output_mask {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    int8_t * output_mask_out = detail::output_mask_out_ptr(req);
    std::copy_n(req.output_mask, req.n_tokens, output_mask_out);
  }
};

struct set_output_mask_last {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    int8_t * output_mask_out = detail::output_mask_out_ptr(req);
    std::fill_n(output_mask_out, req.n_tokens, static_cast<int8_t>(0));
    output_mask_out[req.n_tokens - 1] = 1;
  }
};

struct count_outputs_total {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    const auto & req = ev.request;
    const int8_t * output_mask_out = detail::output_mask_out_ptr(req);
    int32_t total = 0;
    for (int32_t i = 0; i < req.n_tokens; ++i) {
      total += (output_mask_out[i] != 0);
    }
    ev.ctx.outputs_total = total;
  }
};

struct publish_seq_mask_words {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    *ev.request.seq_mask_words_out = ev.ctx.normalized_seq_mask_words;
  }
};

struct publish_positions_count {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    *ev.request.positions_count_out = ev.ctx.normalized_positions_count;
  }
};

struct publish_outputs_total {
  void operator()(const event::batch_runtime & ev, context &) const noexcept {
    *ev.request.outputs_total_out = ev.ctx.outputs_total;
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    detail::write_error(runtime_ev, runtime_ev.ctx.err);
    runtime_ev.request.on_done(events::batch_done{
      .request = &runtime_ev.request,
    });
  }
};

struct publish_done_noop {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    detail::write_error(runtime_ev, runtime_ev.ctx.err);
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    detail::write_error(runtime_ev, runtime_ev.ctx.err);
    runtime_ev.request.on_error(events::batch_error{
      .err = runtime_ev.ctx.err,
      .request = &runtime_ev.request,
    });
  }
};

struct publish_error_noop {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    detail::write_error(runtime_ev, runtime_ev.ctx.err);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
      if constexpr (requires { ev.request.error_out; }) {
        detail::write_error(ev, ev.ctx.err);
      }
    }
  }
};

inline constexpr begin_batch begin_batch{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_internal_error mark_internal_error{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr normalize_seq_from_masks normalize_seq_from_masks{};
inline constexpr normalize_seq_from_primary_ids normalize_seq_from_primary_ids{};
inline constexpr normalize_seq_default normalize_seq_default{};
inline constexpr copy_positions_stride_three copy_positions_stride_three{};
inline constexpr copy_positions_stride_one copy_positions_stride_one{};
inline constexpr probe_positions_seeded probe_positions_seeded{};
inline constexpr probe_positions_unseeded probe_positions_unseeded{};
inline constexpr generate_positions_seeded generate_positions_seeded{};
inline constexpr generate_positions_unseeded generate_positions_unseeded{};
inline constexpr set_output_mask_all set_output_mask_all{};
inline constexpr copy_output_mask copy_output_mask{};
inline constexpr set_output_mask_last set_output_mask_last{};
inline constexpr count_outputs_total count_outputs_total{};
inline constexpr publish_seq_mask_words publish_seq_mask_words{};
inline constexpr publish_positions_count publish_positions_count{};
inline constexpr publish_outputs_total publish_outputs_total{};
inline constexpr publish_done publish_done{};
inline constexpr publish_done_noop publish_done_noop{};
inline constexpr publish_error publish_error{};
inline constexpr publish_error_noop publish_error_noop{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::token::batcher::action
