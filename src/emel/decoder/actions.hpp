#pragma once

#include <algorithm>

#include "emel/callback.hpp"
#include "emel/decoder/context.hpp"

namespace emel::decoder::action {

namespace detail {

inline int32_t normalize_error(const bool ok, const int32_t err) noexcept {
  if (ok && err == EMEL_OK) {
    return EMEL_OK;
  }
  if (err != EMEL_OK) {
    return err;
  }
  return EMEL_ERR_BACKEND;
}

inline int32_t normalize_ubatch_error(const bool ok, const int32_t err) noexcept {
  if (ok && err == EMEL_OK) {
    return EMEL_OK;
  }
  if (err == EMEL_OK || err == EMEL_ERR_INVALID_ARGUMENT) {
    return EMEL_ERR_BACKEND;
  }
  return err;
}

inline int32_t primary_seq_from_mask(
    const uint64_t * seq_masks,
    const int32_t seq_mask_words,
    const int32_t token_idx) noexcept {
  if (seq_masks == nullptr || seq_mask_words <= 0) {
    return 0;
  }
  const size_t base =
      static_cast<size_t>(token_idx) * static_cast<size_t>(seq_mask_words);
  for (int32_t w = 0; w < seq_mask_words; ++w) {
    const uint64_t mask = seq_masks[base + static_cast<size_t>(w)];
    if (mask == 0) {
      continue;
    }
    const int32_t bit = __builtin_ctzll(mask);
    return w * 64 + bit;
  }
  return 0;
}

inline constexpr int32_t k_max_sequences = emel::batch::planner::action::MAX_SEQ;

inline bool has_seq_masks(const context & ctx) noexcept {
  return ctx.seq_masks != nullptr && ctx.seq_masks_count >= ctx.n_tokens &&
         ctx.seq_mask_words > 0 &&
         ctx.seq_mask_words <= emel::batch::planner::action::SEQ_WORDS;
}

inline bool has_seq_primary_ids(const context & ctx) noexcept {
  return ctx.seq_primary_ids != nullptr && ctx.seq_primary_ids_count >= ctx.n_tokens;
}

inline bool resolve_ubatch_span(const context & ctx, const int32_t ubatch_index,
                                int32_t & start, int32_t & end) noexcept {
  if (ubatch_index < 0 || ubatch_index >= ctx.ubatches_total || ctx.n_tokens <= 0 ||
      ctx.token_indices_count < 0 || ctx.token_indices_count > ctx.n_tokens) {
    return false;
  }

  start = ctx.ubatch_token_offsets[ubatch_index];
  end = ctx.ubatch_token_offsets[ubatch_index + 1];
  if (start < 0 || end < start || end > ctx.token_indices_count) {
    return false;
  }
  return end - start == ctx.ubatch_sizes[ubatch_index];
}

inline int32_t resolve_token_sequence_id(const context & ctx, const int32_t token_idx,
                                         int32_t & seq_id) noexcept {
  if (token_idx < 0 || token_idx >= ctx.n_tokens) {
    return EMEL_ERR_BACKEND;
  }

  if (has_seq_primary_ids(ctx)) {
    seq_id = ctx.seq_primary_ids[token_idx];
  } else if (has_seq_masks(ctx)) {
    seq_id = primary_seq_from_mask(ctx.seq_masks, ctx.seq_mask_words, token_idx);
  } else {
    seq_id = 0;
  }

  return seq_id < 0 || seq_id >= k_max_sequences ? EMEL_ERR_INVALID_ARGUMENT : EMEL_OK;
}

inline int32_t collect_ubatch_sequence_counts(
    const context & ctx, const int32_t ubatch_index,
    std::array<int32_t, static_cast<size_t>(k_max_sequences)> & sequence_ids,
    std::array<int32_t, static_cast<size_t>(k_max_sequences)> & sequence_token_counts,
    int32_t & sequence_count) noexcept {
  sequence_ids.fill(0);
  sequence_token_counts.fill(0);
  sequence_count = 0;

  int32_t token_start = 0;
  int32_t token_end = 0;
  if (!resolve_ubatch_span(ctx, ubatch_index, token_start, token_end)) {
    return EMEL_ERR_BACKEND;
  }

  for (int32_t t = token_start; t < token_end; ++t) {
    const int32_t token_idx = ctx.ubatch_token_indices[t];
    int32_t seq_id = 0;
    const int32_t seq_err = resolve_token_sequence_id(ctx, token_idx, seq_id);
    if (seq_err != EMEL_OK) {
      return seq_err;
    }

    const size_t seq_slot = static_cast<size_t>(seq_id);
    if (sequence_token_counts[seq_slot] == 0) {
      if (sequence_count >= k_max_sequences) {
        return EMEL_ERR_BACKEND;
      }
      sequence_ids[static_cast<size_t>(sequence_count)] = seq_id;
      sequence_count += 1;
    }
    sequence_token_counts[seq_slot] += 1;
  }

  if (token_end > token_start && sequence_count <= 0) {
    return EMEL_ERR_BACKEND;
  }
  return EMEL_OK;
}

inline int32_t allocate_ubatch_slots(const int32_t ubatch_index, context & ctx) noexcept {
  if (ctx.memory_coordinator == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }

  std::array<int32_t, static_cast<size_t>(k_max_sequences)> sequence_ids = {};
  std::array<int32_t, static_cast<size_t>(k_max_sequences)> sequence_token_counts = {};
  int32_t sequence_count = 0;
  const int32_t collect_err = collect_ubatch_sequence_counts(
      ctx, ubatch_index, sequence_ids, sequence_token_counts, sequence_count);
  if (collect_err != EMEL_OK) {
    return collect_err;
  }

  for (int32_t i = 0; i < sequence_count; ++i) {
    const int32_t seq_id = sequence_ids[static_cast<size_t>(i)];
    const int32_t token_count = sequence_token_counts[static_cast<size_t>(seq_id)];
    if (token_count <= 0) {
      return EMEL_ERR_BACKEND;
    }

    int32_t alloc_error = EMEL_OK;
    const bool sequence_ok =
        ctx.memory_coordinator->process_event(emel::memory::coordinator::event::allocate_sequence{
          .seq_id = seq_id,
          .error_out = &alloc_error,
        });
    const int32_t normalized_sequence = normalize_error(sequence_ok, alloc_error);
    if (normalized_sequence != EMEL_OK) {
      return normalized_sequence;
    }

    const bool slots_ok =
        ctx.memory_coordinator->process_event(emel::memory::coordinator::event::allocate_slots{
          .seq_id = seq_id,
          .token_count = token_count,
          .error_out = &alloc_error,
        });
    const int32_t normalized_slots = normalize_error(slots_ok, alloc_error);
    if (normalized_slots != EMEL_OK) {
      return normalized_slots;
    }
  }

  return EMEL_OK;
}

inline int32_t rollback_ubatch_slots(const int32_t ubatch_index, context & ctx) noexcept {
  if (ctx.memory_coordinator == nullptr) {
    return EMEL_ERR_BACKEND;
  }

  std::array<int32_t, static_cast<size_t>(k_max_sequences)> sequence_ids = {};
  std::array<int32_t, static_cast<size_t>(k_max_sequences)> sequence_token_counts = {};
  int32_t sequence_count = 0;
  const int32_t collect_err = collect_ubatch_sequence_counts(
      ctx, ubatch_index, sequence_ids, sequence_token_counts, sequence_count);
  if (collect_err != EMEL_OK) {
    return collect_err;
  }

  for (int32_t i = 0; i < sequence_count; ++i) {
    const int32_t seq_id = sequence_ids[static_cast<size_t>(i)];
    const int32_t token_count = sequence_token_counts[static_cast<size_t>(seq_id)];
    if (token_count <= 0) {
      return EMEL_ERR_BACKEND;
    }

    int32_t rollback_error = EMEL_OK;
    const bool rollback_ok =
        ctx.memory_coordinator->process_event(emel::memory::coordinator::event::rollback_slots{
          .seq_id = seq_id,
          .token_count = token_count,
          .error_out = &rollback_error,
        });
    const int32_t normalized = normalize_error(rollback_ok, rollback_error);
    if (normalized != EMEL_OK) {
      return normalized;
    }
  }

  return EMEL_OK;
}

}  // namespace detail

struct begin_decode {
  void operator()(const event::decode & ev, context & ctx) const noexcept {
    ctx.token_ids = ev.token_ids;
    ctx.output_all = ev.output_all;
    ctx.output_mask = ev.output_mask;
    ctx.seq_masks = ev.seq_masks;
    ctx.seq_primary_ids = ev.seq_primary_ids;
    ctx.positions = ev.positions;
    ctx.n_tokens = ev.n_tokens;
    ctx.n_ubatch = ev.n_ubatch;
    ctx.output_mask_count = ev.output_mask_count;
    ctx.seq_mask_words = ev.seq_mask_words;
    ctx.seq_masks_count = ev.seq_masks_count;
    ctx.seq_primary_ids_count = ev.seq_primary_ids_count;
    ctx.positions_count = ev.positions_count;
    ctx.outputs_capacity = ev.outputs_capacity;
    ctx.compute_ctx = ev.compute_ctx;
    ctx.compute_validate = ev.compute_validate;
    ctx.compute_prepare_graph = ev.compute_prepare_graph;
    ctx.compute_alloc_graph = ev.compute_alloc_graph;
    ctx.compute_bind_inputs = ev.compute_bind_inputs;
    ctx.compute_run_backend = ev.compute_run_backend;
    ctx.compute_extract_outputs = ev.compute_extract_outputs;

    ctx.outputs_total = 0;
    ctx.outputs_processed = 0;
    ctx.ubatches_total = 0;
    ctx.ubatches_processed = 0;
    ctx.ubatch_sizes.fill(0);
    ctx.ubatch_seq_ids.fill(0);
    ctx.ubatch_token_indices.fill(0);
    ctx.ubatch_token_offsets.fill(0);
    ctx.ubatch_outputs.fill(0);
    ctx.ubatch_positions.fill(0);
    ctx.ubatch_seq_masks.fill(0);
    ctx.ubatch_seq_primary_ids.fill(0);
    ctx.batched_seq_primary_ids.fill(0);
    ctx.batched_seq_masks.fill(0);
    ctx.batched_positions.fill(0);
    ctx.batched_output_mask.fill(0);
    ctx.batched_outputs_total = 0;
    ctx.batched_positions_count = 0;
    ctx.batched_seq_mask_words = 1;
    ctx.token_indices_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.ubatch_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.phase_retryable = false;
    ctx.rollback_needed = false;

    if (ctx.memory_coordinator != nullptr) {
      ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);
    }
  }
};

struct run_validate {
  void operator()(const event::validate & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;  // GCOVR_EXCL_LINE
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
  }
};

struct run_batch_tokens {
  void operator()(const event::batch_tokens & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    if (ctx.token_batcher == nullptr) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      return;
    }

    const bool ok = ctx.token_batcher->process_event(
        emel::token::batcher::event::batch{
          .token_ids = ctx.token_ids,
          .n_tokens = ctx.n_tokens,
          .seq_masks = ctx.seq_masks,
          .seq_mask_words = ctx.seq_mask_words,
          .seq_masks_count = ctx.seq_masks_count,
          .seq_primary_ids = ctx.seq_primary_ids,
          .seq_primary_ids_count = ctx.seq_primary_ids_count,
          .positions = ctx.positions,
          .positions_count = ctx.positions_count,
          .output_mask = ctx.output_mask,
          .output_mask_count = ctx.output_mask_count,
          .output_all = ctx.output_all,
          .enforce_single_output_per_seq = false,
          .seq_primary_ids_out = ctx.batched_seq_primary_ids.data(),
          .seq_primary_ids_capacity = static_cast<int32_t>(ctx.batched_seq_primary_ids.size()),
          .seq_masks_out = ctx.batched_seq_masks.data(),
          .seq_masks_capacity = static_cast<int32_t>(ctx.batched_seq_masks.size()),
          .positions_out = ctx.batched_positions.data(),
          .positions_capacity = static_cast<int32_t>(ctx.batched_positions.size()),
          .output_mask_out = ctx.batched_output_mask.data(),
          .output_mask_capacity = static_cast<int32_t>(ctx.batched_output_mask.size()),
          .outputs_total_out = &ctx.batched_outputs_total,
          .seq_mask_words_out = &ctx.batched_seq_mask_words,
          .positions_count_out = &ctx.batched_positions_count,
          .error_out = &ctx.phase_error,
        });

    if (!ok || ctx.phase_error != EMEL_OK) {
      *ev.error_out = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
      ctx.phase_error = *ev.error_out;
      return;
    }

    ctx.seq_masks = ctx.batched_seq_masks.data();
    ctx.seq_masks_count = ctx.n_tokens;
    ctx.seq_mask_words = ctx.batched_seq_mask_words;
    ctx.seq_primary_ids = ctx.batched_seq_primary_ids.data();
    ctx.seq_primary_ids_count = ctx.n_tokens;
    ctx.positions = ctx.batched_positions.data();
    ctx.positions_count = ctx.batched_positions_count;
    ctx.output_mask = ctx.batched_output_mask.data();
    ctx.output_mask_count = ctx.n_tokens;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::batch_tokens batch_request{
      .error_out = &ctx.phase_error,
    };
    (*this)(batch_request, ctx);
  }
};

struct reject_invalid_validate {
  void operator()(const event::validate & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }
};

struct run_initialize_batch {
  void operator()(const event::initialize_batch & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    struct plan_reply {
      int32_t * ubatch_sizes_out = nullptr;
      int32_t ubatch_sizes_capacity = 0;
      int32_t * token_indices_out = nullptr;
      int32_t token_indices_capacity = 0;
      int32_t * token_offsets_out = nullptr;
      int32_t token_offsets_capacity = 0;
      int32_t ubatch_count = 0;
      int32_t total_outputs = 0;
      int32_t token_indices_count = 0;
      int32_t token_offsets_count = 0;
      int32_t err = EMEL_OK;

      void on_done(const emel::batch::planner::events::plan_done & reply) noexcept {
        err = EMEL_OK;
        ubatch_count = reply.ubatch_count;
        total_outputs = reply.total_outputs;
        token_indices_count = reply.ubatch_token_indices_count;
        token_offsets_count = reply.ubatch_token_offsets_count;
        if (ubatch_sizes_out == nullptr) {
          return;  // GCOVR_EXCL_LINE
        }
        if (ubatch_sizes_capacity < reply.ubatch_count || reply.ubatch_sizes == nullptr) {
          err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
          return;  // GCOVR_EXCL_LINE
        }
        if (token_indices_out == nullptr || token_offsets_out == nullptr) {
          err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
          return;  // GCOVR_EXCL_LINE
        }
        if (token_indices_capacity < reply.ubatch_token_indices_count ||
            reply.ubatch_token_indices == nullptr) {
          err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
          return;  // GCOVR_EXCL_LINE
        }
        if (token_offsets_capacity < reply.ubatch_token_offsets_count ||
            reply.ubatch_token_offsets == nullptr) {
          err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
          return;  // GCOVR_EXCL_LINE
        }
        for (int32_t i = 0; i < reply.ubatch_count; ++i) {
          ubatch_sizes_out[i] = reply.ubatch_sizes[i];
        }
        for (int32_t i = 0; i < reply.ubatch_token_indices_count; ++i) {
          token_indices_out[i] = reply.ubatch_token_indices[i];
        }
        for (int32_t i = 0; i < reply.ubatch_token_offsets_count; ++i) {
          token_offsets_out[i] = reply.ubatch_token_offsets[i];
        }
      }

      void on_error(const emel::batch::planner::events::plan_error & reply) noexcept {
        err = reply.err;
      }
    };

    plan_reply reply{
      .ubatch_sizes_out = ctx.ubatch_sizes.data(),
      .ubatch_sizes_capacity = static_cast<int32_t>(ctx.ubatch_sizes.size()),
      .token_indices_out = ctx.ubatch_token_indices.data(),
      .token_indices_capacity = static_cast<int32_t>(ctx.ubatch_token_indices.size()),
      .token_offsets_out = ctx.ubatch_token_offsets.data(),
      .token_offsets_capacity = static_cast<int32_t>(ctx.ubatch_token_offsets.size()),
    };

    const auto on_done =
        emel::callback<void(const emel::batch::planner::events::plan_done &)>::from<
            plan_reply, &plan_reply::on_done>(&reply);
    const auto on_error =
        emel::callback<void(const emel::batch::planner::events::plan_error &)>::from<
            plan_reply, &plan_reply::on_error>(&reply);

    const emel::batch::planner::event::plan_mode plan_mode =
        ctx.output_all ? emel::batch::planner::event::plan_mode::seq
                       : emel::batch::planner::event::plan_mode::equal;
    const bool ok = ctx.batch_planner->process_event(emel::batch::planner::event::plan{
      .token_ids = ctx.token_ids,
      .n_tokens = ctx.n_tokens,
      .n_ubatch = ctx.n_ubatch,
      .mode = plan_mode,
      .seq_masks = ctx.seq_masks,
      .seq_masks_count = ctx.seq_masks_count,
      .seq_primary_ids = ctx.seq_primary_ids,
      .seq_primary_ids_count = ctx.seq_primary_ids_count,
      .equal_sequential = true,
      .seq_mask_words = ctx.seq_mask_words,
      .output_mask = ctx.output_mask,
      .output_mask_count = ctx.output_mask_count,
      .output_all = ctx.output_all,
      .on_done = on_done,
      .on_error = on_error,
    });

    if (!ok || reply.err != EMEL_OK) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      return;
    }

    if (reply.ubatch_count <= 0 || reply.total_outputs < 0) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }
    if (reply.token_indices_count != ctx.n_tokens) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }
    if (reply.token_offsets_count != reply.ubatch_count + 1) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    ctx.ubatches_total = reply.ubatch_count;
    ctx.ubatches_processed = 0;
    ctx.outputs_processed = 0;
    ctx.token_indices_count = reply.token_indices_count;

    int32_t outputs_total = 0;
    const bool use_output_mask =
        !ctx.output_all && ctx.output_mask != nullptr && ctx.output_mask_count >= ctx.n_tokens;
    const bool output_last_only = !ctx.output_all && ctx.output_mask == nullptr;
    const bool has_seq_masks =
        ctx.seq_masks != nullptr && ctx.seq_masks_count >= ctx.n_tokens;
    const bool has_seq_ids =
        ctx.seq_primary_ids != nullptr && ctx.seq_primary_ids_count >= ctx.n_tokens;
    const int32_t last_token = ctx.n_tokens - 1;

    for (int32_t i = 0; i < ctx.ubatches_total; ++i) {
      const int32_t size = ctx.ubatch_sizes[i];
      const int32_t start = ctx.ubatch_token_offsets[i];
      const int32_t end = ctx.ubatch_token_offsets[i + 1];
      if (size <= 0 || start < 0 || end < start || end > ctx.token_indices_count) {
        *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
        ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
        return;  // GCOVR_EXCL_LINE
      }
      if (end - start != size) {
        *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
        ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
        return;  // GCOVR_EXCL_LINE
      }

      int32_t outputs = 0;
      for (int32_t t = start; t < end; ++t) {
        const int32_t token_idx = ctx.ubatch_token_indices[t];
        if (token_idx < 0 || token_idx >= ctx.n_tokens) {
          *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
          ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
          return;  // GCOVR_EXCL_LINE
        }
        if (ctx.output_all) {
          outputs += 1;
        } else if (use_output_mask) {
          outputs += (ctx.output_mask[token_idx] != 0);
        } else if (output_last_only) {
          outputs += (token_idx == last_token);
        }
      }
      ctx.ubatch_outputs[i] = outputs;
      outputs_total += outputs;

      const int32_t first_token = ctx.ubatch_token_indices[start];
      int32_t seq_id = 0;
      if (has_seq_ids) {
        seq_id = ctx.seq_primary_ids[first_token];
      } else if (has_seq_masks) {
        seq_id = detail::primary_seq_from_mask(ctx.seq_masks, ctx.seq_mask_words, first_token);
      }
      ctx.ubatch_seq_ids[i] = seq_id;
    }

    if (outputs_total != reply.total_outputs) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    ctx.outputs_total = outputs_total;
    if (ctx.n_ubatch <= 0) {
      ctx.n_ubatch = ctx.n_tokens;
    }
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::initialize_batch initialize{
      .error_out = &ctx.phase_error,
    };
    (*this)(initialize, ctx);
  }
};

struct run_update_memory {
  void operator()(const event::update_memory & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    if (ctx.memory_coordinator == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    if (!ctx.memory_reserved) {
      int32_t reserve_error = EMEL_OK;
      const bool reserved = ctx.memory_coordinator->process_event(emel::memory::coordinator::event::reserve{
        .max_sequences = 256,
        .max_blocks = 32768,
        .block_tokens = 16,
        .error_out = &reserve_error,
      });
      const int32_t normalized = detail::normalize_error(reserved, reserve_error);
      if (normalized != EMEL_OK) {
        *ev.error_out = normalized;
        ctx.phase_error = normalized;
        return;
      }
      ctx.memory_reserved = true;
    }
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::update_memory update{
      .error_out = &ctx.phase_error,
    };
    (*this)(update, ctx);
  }
};

struct run_prepare_memory_batch {
  void operator()(const event::prepare_memory_batch & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
    ctx.phase_retryable = false;
    if (ev.retryable_out != nullptr) {
      *ev.retryable_out = false;
    }

    if (ctx.memory_coordinator == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    for (int32_t i = 0; i < ctx.ubatches_total; ++i) {
      const int32_t normalized = detail::allocate_ubatch_slots(i, ctx);
      if (normalized != EMEL_OK) {
        *ev.error_out = normalized;
        ctx.phase_error = normalized;
        ctx.phase_retryable = false;
        return;
      }
    }
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.phase_retryable = false;
    event::prepare_memory_batch prepare{
      .error_out = &ctx.phase_error,
      .retryable_out = &ctx.phase_retryable,
    };
    (*this)(prepare, ctx);
  }
};

struct run_optimize_memory {
  void operator()(const event::optimize_memory & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    (void)ctx;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::optimize_memory optimize{
      .error_out = &ctx.phase_error,
    };
    (*this)(optimize, ctx);
  }
};

struct run_reserve_output {
  void operator()(const event::reserve_output & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
    if (ctx.outputs_capacity > 0 && ctx.outputs_total > ctx.outputs_capacity) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
    }
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
  }
};

struct reject_invalid_reserve_output {
  void operator()(const event::reserve_output & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_ERR_BACKEND;
    ctx.phase_error = EMEL_ERR_BACKEND;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
  }
};

struct run_process_ubatch {
  void operator()(const event::process_ubatch & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    if (ev.rollback_needed_out != nullptr) {
      *ev.rollback_needed_out = false;
    }
    ctx.phase_error = EMEL_OK;
    ctx.rollback_needed = false;

    if (ctx.ubatch_executor == nullptr) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      ctx.ubatch_error = EMEL_ERR_BACKEND;
      ctx.rollback_needed = true;
      if (ev.rollback_needed_out != nullptr) {
        *ev.rollback_needed_out = true;
      }
      return;
    }
    if (ctx.memory_coordinator == nullptr) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      ctx.ubatch_error = EMEL_ERR_BACKEND;
      return;
    }

    if (ctx.ubatches_processed < 0 || ctx.ubatches_processed >= ctx.ubatches_total) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      ctx.ubatch_error = EMEL_ERR_BACKEND;
      return;
    }

    const int32_t current = ctx.ubatch_sizes[ctx.ubatches_processed];
    const int32_t expected_outputs = ctx.ubatch_outputs[ctx.ubatches_processed];
    const int32_t token_start = ctx.ubatch_token_offsets[ctx.ubatches_processed];
    const int32_t token_end = ctx.ubatch_token_offsets[ctx.ubatches_processed + 1];
    if (token_start < 0 || token_end < token_start ||
        token_end > ctx.token_indices_count || token_end - token_start != current) {
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.ubatch_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    const bool has_seq_masks =
        ctx.seq_masks != nullptr && ctx.seq_masks_count >= ctx.n_tokens;
    const bool has_seq_ids =
        ctx.seq_primary_ids != nullptr && ctx.seq_primary_ids_count >= ctx.n_tokens;

    int32_t pos_stride = 0;
    if (ctx.positions != nullptr) {
      if (ctx.positions_count >= ctx.n_tokens * 3) {
        pos_stride = 3;
      } else if (ctx.positions_count >= ctx.n_tokens) {
        pos_stride = 1;
      }
    }

    const int32_t * positions = nullptr;
    int32_t positions_count = 0;
    if (pos_stride > 0) {
      if (pos_stride == 1) {
        for (int32_t i = 0; i < current; ++i) {
          const int32_t token_idx = ctx.ubatch_token_indices[token_start + i];
          ctx.ubatch_positions[i] = ctx.positions[token_idx];
        }
      } else {
        for (int32_t i = 0; i < current; ++i) {
          const int32_t token_idx = ctx.ubatch_token_indices[token_start + i];
          ctx.ubatch_positions[i] = ctx.positions[token_idx];
          ctx.ubatch_positions[i + current] = ctx.positions[ctx.n_tokens + token_idx];
          ctx.ubatch_positions[i + current * 2] =
              ctx.positions[ctx.n_tokens * 2 + token_idx];
        }
      }
      positions = ctx.ubatch_positions.data();
      positions_count = current * pos_stride;
    }

    const uint64_t * seq_masks = nullptr;
    int32_t seq_mask_words = 0;
    int32_t seq_masks_count = 0;
    if (has_seq_masks) {
      seq_mask_words = ctx.seq_mask_words;
      for (int32_t i = 0; i < current; ++i) {
        const int32_t token_idx = ctx.ubatch_token_indices[token_start + i];
        const size_t src_base =
            static_cast<size_t>(token_idx) * static_cast<size_t>(seq_mask_words);
        const size_t dst_base =
            static_cast<size_t>(i) * static_cast<size_t>(seq_mask_words);
        for (int32_t w = 0; w < seq_mask_words; ++w) {
          ctx.ubatch_seq_masks[dst_base + static_cast<size_t>(w)] =
              ctx.seq_masks[src_base + static_cast<size_t>(w)];
        }
      }
      seq_masks = ctx.ubatch_seq_masks.data();
      seq_masks_count = current;
    }

    const int32_t * seq_primary_ids = nullptr;
    int32_t seq_primary_ids_count = 0;
    if (has_seq_ids) {
      for (int32_t i = 0; i < current; ++i) {
        const int32_t token_idx = ctx.ubatch_token_indices[token_start + i];
        ctx.ubatch_seq_primary_ids[i] = ctx.seq_primary_ids[token_idx];
      }
      seq_primary_ids = ctx.ubatch_seq_primary_ids.data();
      seq_primary_ids_count = current;
    }
    int32_t produced = 0;
    int32_t kv_tokens = 0;
    bool rollback_attempted = false;
    int32_t ubatch_error = EMEL_OK;
    const bool ok = ctx.ubatch_executor->process_event(emel::graph::processor::event::execute{
      .ubatch_index = ctx.ubatches_processed,
      .ubatch_size = current,
      .memory_sm = ctx.memory_coordinator.get(),
      .memory_view = ctx.memory_coordinator->view(),
      .expected_outputs = expected_outputs,
      .compute_ctx = ctx.compute_ctx,
      .positions = positions,
      .positions_count = positions_count,
      .seq_masks = seq_masks,
      .seq_mask_words = seq_mask_words,
      .seq_masks_count = seq_masks_count,
      .seq_primary_ids = seq_primary_ids,
      .seq_primary_ids_count = seq_primary_ids_count,
      .validate = ctx.compute_validate,
      .prepare_graph = ctx.compute_prepare_graph,
      .alloc_graph = ctx.compute_alloc_graph,
      .bind_inputs = ctx.compute_bind_inputs,
      .run_backend = ctx.compute_run_backend,
      .extract_outputs = ctx.compute_extract_outputs,
      .outputs_produced_out = &produced,
      .kv_tokens_out = &kv_tokens,
      .rollback_attempted_out = &rollback_attempted,
      .error_out = &ubatch_error,
    });

    const int32_t normalized = detail::normalize_ubatch_error(ok, ubatch_error);
    if (normalized != EMEL_OK) {
      *ev.error_out = normalized;  // GCOVR_EXCL_LINE
      ctx.phase_error = normalized;  // GCOVR_EXCL_LINE
      ctx.ubatch_error = normalized;  // GCOVR_EXCL_LINE
      const bool rollback_needed = !rollback_attempted;
      if (ev.rollback_needed_out != nullptr) {  // GCOVR_EXCL_LINE
        *ev.rollback_needed_out = rollback_needed;  // GCOVR_EXCL_LINE
      }
      ctx.rollback_needed = rollback_needed;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    if (produced < 0) {  // GCOVR_EXCL_LINE
      *ev.error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.phase_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      ctx.ubatch_error = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return;  // GCOVR_EXCL_LINE
    }

    ctx.outputs_processed += produced;
    ctx.ubatches_processed += 1;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.rollback_needed = false;
    event::process_ubatch process{
      .error_out = &ctx.phase_error,
      .rollback_needed_out = &ctx.rollback_needed,
    };
    (*this)(process, ctx);
    if (ctx.phase_error != EMEL_OK) {
      ctx.ubatch_error = ctx.phase_error;
    }
  }
};

struct on_invalid_ubatch_size {
  void operator()(const event::process_ubatch & ev, context & ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_BACKEND;
    }
    if (ev.rollback_needed_out != nullptr) {
      *ev.rollback_needed_out = false;
    }
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.ubatch_error = EMEL_ERR_BACKEND;
    ctx.rollback_needed = false;
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.ubatch_error = EMEL_ERR_BACKEND;
    ctx.rollback_needed = false;
  }
};

struct run_rollback_ubatch {
  void operator()(const event::rollback_ubatch & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;

    if (!ev.rollback_needed) {
      return;  // GCOVR_EXCL_LINE
    }

    if (ctx.memory_coordinator == nullptr || ctx.ubatches_total <= 0) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      return;
    }

    int32_t rollback_start = ctx.ubatches_processed;
    if (rollback_start >= ctx.ubatches_total) {
      rollback_start = ctx.ubatches_total - 1;
    }
    if (rollback_start < 0 || rollback_start >= ctx.ubatches_total) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
      return;
    }

    for (int32_t i = rollback_start; i < ctx.ubatches_total; ++i) {
      const int32_t normalized = detail::rollback_ubatch_slots(i, ctx);
      if (normalized != EMEL_OK) {
        *ev.error_out = normalized;
        ctx.phase_error = normalized;
        return;  // GCOVR_EXCL_LINE
      }
    }

    if (ctx.outputs_processed > ctx.outputs_total) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
    }
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::rollback_ubatch rollback{
      .error_out = &ctx.phase_error,
      .rollback_needed = ctx.rollback_needed,
    };
    (*this)(rollback, ctx);
  }
};

struct run_finalize_outputs {
  void operator()(const event::finalize_outputs & ev, context & ctx) const noexcept {
    if (ev.error_out == nullptr) {
      return;
    }
    *ev.error_out = EMEL_OK;
    ctx.phase_error = EMEL_OK;
    if (ctx.outputs_processed != ctx.outputs_total) {
      *ev.error_out = EMEL_ERR_BACKEND;
      ctx.phase_error = EMEL_ERR_BACKEND;
    }
  }

  template <class ev>
  void operator()(const ev &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    event::finalize_outputs finalize{
      .error_out = &ctx.phase_error,
    };
    (*this)(finalize, ctx);
  }
};

struct dispatch_decoding_done_to_owner {
  void operator()(const events::decoding_done & ev, context &) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    if (ev.dispatch_event != nullptr) {
      (void)ev.dispatch_event(ev.owner_sm, events::owner_event{
                                               .type = events::owner_event::kind::done,
                                               .done = ev,
                                           });
    }
  }
};

struct dispatch_decoding_error_to_owner {
  void operator()(const events::decoding_error & ev, context &) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = ev.err == EMEL_OK ? EMEL_ERR_BACKEND : ev.err;
    }
    if (ev.dispatch_event != nullptr) {
      (void)ev.dispatch_event(ev.owner_sm, events::owner_event{
                                               .type = events::owner_event::kind::error,
                                               .error = ev,
                                           });
    }
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
  }
};

struct capture_rollback_error {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct capture_ubatch_error {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = ctx.ubatch_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.ubatch_error;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct on_unexpected {
  template <class event>
  void operator()(const event & ev, context & ctx) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
    ctx.phase_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr begin_decode begin_decode{};
inline constexpr run_validate run_validate{};
inline constexpr run_batch_tokens run_batch_tokens{};
inline constexpr reject_invalid_validate reject_invalid_validate{};
inline constexpr run_initialize_batch run_initialize_batch{};
inline constexpr run_update_memory run_update_memory{};
inline constexpr run_prepare_memory_batch run_prepare_memory_batch{};
inline constexpr run_optimize_memory run_optimize_memory{};
inline constexpr run_reserve_output run_reserve_output{};
inline constexpr reject_invalid_reserve_output reject_invalid_reserve_output{};
inline constexpr run_process_ubatch run_process_ubatch{};
inline constexpr on_invalid_ubatch_size on_invalid_ubatch_size{};
inline constexpr run_rollback_ubatch run_rollback_ubatch{};
inline constexpr run_finalize_outputs run_finalize_outputs{};
inline constexpr dispatch_decoding_done_to_owner dispatch_decoding_done_to_owner{};
inline constexpr dispatch_decoding_error_to_owner dispatch_decoding_error_to_owner{};
inline constexpr mark_done mark_done{};
inline constexpr capture_rollback_error capture_rollback_error{};
inline constexpr capture_ubatch_error capture_ubatch_error{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::decoder::action
