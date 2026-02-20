#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include "emel/batch/sanitizer/context.hpp"

namespace emel::batch::sanitizer::action {

namespace detail {

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
  if (ctx.error_out != nullptr) {
    *ctx.error_out = err;
  }
}

inline void write_outputs_total(context & ctx, const int32_t value) noexcept {
  if (ctx.outputs_total_out != nullptr) {
    *ctx.outputs_total_out = value;
  }
}

inline void write_seq_mask_words(context & ctx, const int32_t value) noexcept {
  if (ctx.seq_mask_words_out != nullptr) {
    *ctx.seq_mask_words_out = value;
  }
}

inline void write_positions_count(context & ctx, const int32_t value) noexcept {
  if (ctx.positions_count_out != nullptr) {
    *ctx.positions_count_out = value;
  }
}

inline int32_t mask_primary_id(const uint64_t * mask, const int32_t words) noexcept {
  for (int32_t w = 0; w < words; ++w) {
    const uint64_t bits = mask[static_cast<size_t>(w)];
    if (bits == 0) {
      continue;
    }
    const int32_t bit = __builtin_ctzll(bits);
    return w * 64 + bit;
  }
  return -1;
}

inline bool mask_has_bit(const uint64_t * mask,
                         const int32_t words,
                         const int32_t seq_id) noexcept {
  if (seq_id < 0) {
    return false;
  }
  const int32_t word = seq_id / 64;
  if (word < 0 || word >= words) {
    return false;
  }
  const int32_t bit = seq_id % 64;
  return (mask[static_cast<size_t>(word)] & (1ULL << bit)) != 0;
}

inline void clear_mask(uint64_t * mask, const int32_t words) noexcept {
  for (int32_t w = 0; w < words; ++w) {
    mask[static_cast<size_t>(w)] = 0;
  }
}

inline void set_mask_bit(uint64_t * mask,
                         const int32_t words,
                         const int32_t seq_id) noexcept {
  const int32_t word = seq_id / 64;
  if (word < 0 || word >= words) {
    return;
  }
  const int32_t bit = seq_id % 64;
  mask[static_cast<size_t>(word)] |= (1ULL << bit);
}

inline bool mask_empty(const uint64_t * mask, const int32_t words) noexcept {
  for (int32_t w = 0; w < words; ++w) {
    if (mask[static_cast<size_t>(w)] != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace detail

struct begin_sanitize {
  void operator()(const event::sanitize_decode & ev, context & ctx) const noexcept {
    ctx.token_ids = ev.token_ids;
    ctx.n_tokens = ev.n_tokens;
    ctx.seq_masks = ev.seq_masks;
    ctx.seq_mask_words = ev.seq_mask_words;
    ctx.seq_masks_count = ev.seq_masks_count;
    ctx.seq_primary_ids = ev.seq_primary_ids;
    ctx.seq_primary_ids_count = ev.seq_primary_ids_count;
    ctx.positions = ev.positions;
    ctx.positions_count = ev.positions_count;
    ctx.output_mask = ev.output_mask;
    ctx.output_mask_count = ev.output_mask_count;
    ctx.output_all = ev.output_all;
    ctx.enforce_single_output_per_seq = ev.enforce_single_output_per_seq;

    ctx.seq_primary_ids_out = ev.seq_primary_ids_out;
    ctx.seq_primary_ids_capacity = ev.seq_primary_ids_capacity;
    ctx.seq_masks_out = ev.seq_masks_out;
    ctx.seq_masks_capacity = ev.seq_masks_capacity;
    ctx.positions_out = ev.positions_out;
    ctx.positions_capacity = ev.positions_capacity;
    ctx.output_mask_out = ev.output_mask_out;
    ctx.output_mask_capacity = ev.output_mask_capacity;
    ctx.outputs_total_out = ev.outputs_total_out;
    ctx.seq_mask_words_out = ev.seq_mask_words_out;
    ctx.positions_count_out = ev.positions_count_out;
    ctx.error_out = ev.error_out;

    ctx.outputs_total = 0;
    ctx.normalized_seq_mask_words = 1;
    ctx.normalized_positions_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ctx.error_out != nullptr) {
      *ctx.error_out = EMEL_OK;
    }
    detail::write_outputs_total(ctx, 0);
  }
};

struct reject_invalid_sanitize {
  void operator()(const event::sanitize_decode & ev, context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
  }
};

struct run_sanitize_decode {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;

    if (ctx.n_tokens <= 0 || ctx.token_ids == nullptr) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (ctx.n_tokens > MAX_TOKENS) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (ctx.seq_primary_ids_out == nullptr || ctx.seq_primary_ids_capacity < ctx.n_tokens) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (ctx.seq_masks_out == nullptr || ctx.seq_masks_capacity <= 0) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (ctx.positions_out == nullptr || ctx.positions_capacity < ctx.n_tokens) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (ctx.output_mask_out == nullptr || ctx.output_mask_capacity < ctx.n_tokens) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    const bool has_seq_masks_in =
        ctx.seq_masks != nullptr && ctx.seq_masks_count >= ctx.n_tokens;
    const bool has_seq_primary_in =
        ctx.seq_primary_ids != nullptr && ctx.seq_primary_ids_count >= ctx.n_tokens;

    int32_t mask_words = ctx.seq_mask_words;
    if (!has_seq_masks_in) {
      mask_words = 1;
    } else if (mask_words <= 0 || mask_words > SEQ_WORDS) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    if (ctx.seq_masks_capacity < ctx.n_tokens * mask_words) {
      detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    ctx.normalized_seq_mask_words = mask_words;
    detail::write_seq_mask_words(ctx, mask_words);

    const bool has_positions_in = ctx.positions != nullptr;
    int32_t pos_stride = 0;
    if (has_positions_in) {
      if (ctx.positions_count >= ctx.n_tokens * 3) {
        pos_stride = 3;
      } else if (ctx.positions_count >= ctx.n_tokens) {
        pos_stride = 1;
      } else {
        detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }
    }

    if (pos_stride == 3) {
      if (ctx.positions_capacity < ctx.n_tokens * 3) {
        detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }
      ctx.normalized_positions_count = ctx.n_tokens * 3;
    } else {
      ctx.normalized_positions_count = ctx.n_tokens;
    }
    detail::write_positions_count(ctx, ctx.normalized_positions_count);

    const bool has_output_mask_in =
        ctx.output_mask != nullptr && ctx.output_mask_count >= ctx.n_tokens;

    // Normalize seq masks + primary ids
    for (int32_t i = 0; i < ctx.n_tokens; ++i) {
      uint64_t * out_mask = ctx.seq_masks_out + static_cast<size_t>(i) * mask_words;
      detail::clear_mask(out_mask, mask_words);

      if (has_seq_masks_in) {
        const uint64_t * in_mask = ctx.seq_masks + static_cast<size_t>(i) * mask_words;
        for (int32_t w = 0; w < mask_words; ++w) {
          out_mask[static_cast<size_t>(w)] = in_mask[static_cast<size_t>(w)];
        }
      } else if (has_seq_primary_in) {
        const int32_t seq_id = ctx.seq_primary_ids[i];
        if (seq_id < 0 || seq_id >= mask_words * 64) {
          detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
          return;
        }
        detail::set_mask_bit(out_mask, mask_words, seq_id);
      } else {
        detail::set_mask_bit(out_mask, mask_words, 0);
      }

      if (detail::mask_empty(out_mask, mask_words)) {
        detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }

      const int32_t primary = detail::mask_primary_id(out_mask, mask_words);
      if (primary < 0) {
        detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
        return;
      }

      if (has_seq_primary_in && has_seq_masks_in) {
        const int32_t seq_id = ctx.seq_primary_ids[i];
        if (!detail::mask_has_bit(out_mask, mask_words, seq_id)) {
          detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
          return;
        }
      }

      ctx.seq_primary_ids_out[i] = primary;
    }

    // Normalize positions
    if (has_positions_in) {
      const int32_t count = ctx.normalized_positions_count;
      std::copy(ctx.positions, ctx.positions + count, ctx.positions_out);
    } else {
      std::array<int32_t, MAX_SEQ> next_pos = {};
      next_pos.fill(0);
      for (int32_t i = 0; i < ctx.n_tokens; ++i) {
        const uint64_t * mask = ctx.seq_masks_out + static_cast<size_t>(i) * mask_words;
        const int32_t primary = ctx.seq_primary_ids_out[i];
        const int32_t pos = next_pos[primary];
        ctx.positions_out[i] = pos;

        for (int32_t w = 0; w < mask_words; ++w) {
          uint64_t bits = mask[static_cast<size_t>(w)];
          while (bits != 0) {
            const int32_t bit = __builtin_ctzll(bits);
            const int32_t seq_id = w * 64 + bit;
            next_pos[seq_id] = pos + 1;
            bits &= (bits - 1);
          }
        }
      }
    }

    // Normalize output mask
    if (ctx.output_all) {
      std::fill_n(ctx.output_mask_out, ctx.n_tokens, static_cast<int8_t>(1));
    } else if (has_output_mask_in) {
      std::copy(ctx.output_mask, ctx.output_mask + ctx.n_tokens, ctx.output_mask_out);
    } else {
      std::fill_n(ctx.output_mask_out, ctx.n_tokens, static_cast<int8_t>(0));
      ctx.output_mask_out[ctx.n_tokens - 1] = 1;
    }

    if (ctx.output_all && has_output_mask_in) {
      bool warn = false;
      for (int32_t i = 0; i < ctx.n_tokens; ++i) {
        if (ctx.output_mask_out[i] == 0) {
          warn = true;
          break;
        }
      }
      if (warn) {
        std::fill_n(ctx.output_mask_out, ctx.n_tokens, static_cast<int8_t>(1));
      }
    }

    ctx.outputs_total = 0;
    for (int32_t i = 0; i < ctx.n_tokens; ++i) {
      ctx.outputs_total += (ctx.output_mask_out[i] != 0);
    }
    detail::write_outputs_total(ctx, ctx.outputs_total);

    // Validate per-sequence output constraint if requested.
    if (ctx.enforce_single_output_per_seq) {
      std::array<int32_t, MAX_SEQ> seq_output_count = {};
      seq_output_count.fill(0);
      for (int32_t i = 0; i < ctx.n_tokens; ++i) {
        if (ctx.output_mask_out[i] == 0) {
          continue;
        }
        const uint64_t * mask = ctx.seq_masks_out + static_cast<size_t>(i) * mask_words;
        for (int32_t w = 0; w < mask_words; ++w) {
          uint64_t bits = mask[static_cast<size_t>(w)];
          while (bits != 0) {
            const int32_t bit = __builtin_ctzll(bits);
            const int32_t seq_id = w * 64 + bit;
            seq_output_count[seq_id] += 1;
            if (seq_output_count[seq_id] > 1) {
              detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
              return;
            }
            bits &= (bits - 1);
          }
        }
      }
    }

    // Continuity and subset checks.
    std::array<int32_t, MAX_SEQ> seq_last_pos = {};
    std::array<int32_t, MAX_SEQ> seq_pos_min = {};
    std::array<int32_t, MAX_SEQ> seq_pos_max = {};
    std::array<int32_t, MAX_SEQ> seq_pos_count = {};
    seq_last_pos.fill(-1);
    seq_pos_min.fill(std::numeric_limits<int32_t>::max());
    seq_pos_max.fill(std::numeric_limits<int32_t>::min());
    seq_pos_count.fill(0);

    std::array<uint64_t, static_cast<size_t>(MAX_SEQ * SEQ_WORDS)> cur_seq_set = {};
    for (int32_t s = 0; s < MAX_SEQ; ++s) {
      for (int32_t w = 0; w < mask_words; ++w) {
        cur_seq_set[static_cast<size_t>(s * mask_words + w)] = ~0ULL;
      }
    }

    for (int32_t i = 0; i < ctx.n_tokens; ++i) {
      const int32_t pos = ctx.positions_out[i];
      const uint64_t * mask = ctx.seq_masks_out + static_cast<size_t>(i) * mask_words;
      for (int32_t w = 0; w < mask_words; ++w) {
        uint64_t bits = mask[static_cast<size_t>(w)];
        while (bits != 0) {
          const int32_t bit = __builtin_ctzll(bits);
          const int32_t seq_id = w * 64 + bit;

          const int32_t last = seq_last_pos[seq_id];
          if (last >= 0 && pos < last) {
            detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
            return;
          }

          if (pos != last) {
            seq_pos_count[seq_id] += 1;
          }
          seq_last_pos[seq_id] = pos;
          if (pos < seq_pos_min[seq_id]) {
            seq_pos_min[seq_id] = pos;
          }
          if (pos > seq_pos_max[seq_id]) {
            seq_pos_max[seq_id] = pos;
          }

          uint64_t * cur_mask =
              cur_seq_set.data() + static_cast<size_t>(seq_id * mask_words);
          for (int32_t mw = 0; mw < mask_words; ++mw) {
            cur_mask[static_cast<size_t>(mw)] &=
                mask[static_cast<size_t>(mw)];
          }
          if (detail::mask_empty(cur_mask, mask_words)) {
            detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
            return;
          }

          bits &= (bits - 1);
        }
      }
    }

    if (pos_stride <= 1) {
      for (int32_t s = 0; s < MAX_SEQ; ++s) {
        if (seq_pos_count[s] == 0) {
          continue;
        }
        const int32_t min_pos = seq_pos_min[s];
        const int32_t max_pos = seq_pos_max[s];
        if (min_pos == std::numeric_limits<int32_t>::max() ||
            max_pos == std::numeric_limits<int32_t>::min()) {
          continue;
        }
        if (max_pos - min_pos + 1 > seq_pos_count[s]) {
          detail::set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
          return;
        }
      }
    }
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
    if (ctx.error_out != nullptr) {
      *ctx.error_out = ctx.last_error;
    }
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context & ctx) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
    detail::set_error(ctx, EMEL_ERR_BACKEND);
  }
};

inline constexpr begin_sanitize begin_sanitize{};
inline constexpr reject_invalid_sanitize reject_invalid_sanitize{};
inline constexpr run_sanitize_decode run_sanitize_decode{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::batch::sanitizer::action
