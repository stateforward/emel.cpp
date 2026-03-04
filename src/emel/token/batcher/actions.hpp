#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>

#include "emel/error/error.hpp"
#include "emel/token/batcher/context.hpp"
#include "emel/token/batcher/errors.hpp"
#include "emel/token/batcher/events.hpp"

namespace emel::token::batcher::detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

enum class probe_status : uint8_t {
  ok = 0u,
  backend_error = 1u,
  invalid = 2u,
};

inline bool has_seq_masks_input(const event::batch & req) noexcept {
  return req.seq_masks != nullptr && req.seq_masks_count >= req.n_tokens;
}

inline bool has_seq_primary_input(const event::batch & req) noexcept {
  return req.seq_primary_ids != nullptr && req.seq_primary_ids_count >= req.n_tokens;
}

inline bool has_output_mask_input(const event::batch & req) noexcept {
  return req.output_mask != nullptr && req.output_mask_count >= req.n_tokens;
}

inline int32_t effective_mask_words(const event::batch & req) noexcept {
  const std::array<int32_t, 2> word_candidates = {1, req.seq_mask_words};
  return word_candidates[static_cast<size_t>(has_seq_masks_input(req))];
}

inline int32_t positions_stride(const event::batch & req) noexcept {
  constexpr std::array<int32_t, 2> stride_one_or_invalid = {-1, 1};
  const bool has_positions = req.positions != nullptr;
  const bool stride_three = req.positions_count >= req.n_tokens * 3;
  const bool stride_one = req.positions_count >= req.n_tokens;
  const int32_t with_positions =
      static_cast<int32_t>(stride_three) * 3 +
      (1 - static_cast<int32_t>(stride_three)) *
          stride_one_or_invalid[static_cast<size_t>(stride_one)];
  return static_cast<int32_t>(has_positions) * with_positions;
}

inline int32_t normalized_positions_count(const event::batch & req) noexcept {
  const size_t is_stride_three = static_cast<size_t>(positions_stride(req) == 3);
  const std::array<int32_t, 2> count_candidates = {req.n_tokens, req.n_tokens * 3};
  return count_candidates[is_stride_three];
}

inline const int32_t * token_ids_ptr(const event::batch & req) noexcept {
  return &req.token_ids;
}

inline int32_t * seq_primary_ids_out_ptr(const event::batch & req) noexcept {
  return &req.seq_primary_ids_out;
}

inline uint64_t * seq_masks_out_ptr(const event::batch & req) noexcept {
  return &req.seq_masks_out;
}

inline int32_t * positions_out_ptr(const event::batch & req) noexcept {
  return &req.positions_out;
}

inline int8_t * output_mask_out_ptr(const event::batch & req) noexcept {
  return &req.output_mask_out;
}

inline void write_error(const event::batch_runtime & ev, const emel::error::type value) noexcept {
  ev.request.error_out = value;
}

inline bool mask_empty(const uint64_t * mask, const int32_t words) noexcept {
  uint64_t combined = 0U;
  for (int32_t w = 0; w < words; ++w) {
    combined |= mask[static_cast<size_t>(w)];
  }
  return combined == 0U;
}

inline void clear_mask(uint64_t * mask, const int32_t words) noexcept {
  for (int32_t w = 0; w < words; ++w) {
    mask[static_cast<size_t>(w)] = 0U;
  }
}

inline void set_mask_bit(uint64_t * mask, const int32_t words, const int32_t seq_id) noexcept {
  const int32_t word = seq_id / 64;
  const uint32_t bit = static_cast<uint32_t>(seq_id) & 63U;
  const bool valid = words > 0 && word >= 0 && word < words;
  while (valid) {
    mask[static_cast<size_t>(word)] |= (uint64_t{1} << bit);
    break;
  }
}

inline bool mask_has_bit(const uint64_t * mask,
                         const int32_t words,
                         const int32_t seq_id) noexcept {
  const bool non_negative = seq_id >= 0;
  const int32_t word = seq_id / 64;
  const bool in_range = word >= 0 && word < words;
  const bool valid = non_negative && in_range;
  const uint32_t bit = static_cast<uint32_t>(seq_id) & 63U;
  return valid && ((mask[static_cast<size_t>(word)] & (uint64_t{1} << bit)) != 0U);
}

inline int32_t mask_primary_id(const uint64_t * mask, const int32_t words) noexcept {
  int32_t w = 0;
  while (w < words && mask[static_cast<size_t>(w)] == 0U) {
    ++w;
  }
  const bool found = w < words;
  int32_t bit = 0;
  while (found) {
    bit = static_cast<int32_t>(std::countr_zero(mask[static_cast<size_t>(w)]));
    break;
  }
  return static_cast<int32_t>(found) * (w * 64 + bit) + static_cast<int32_t>(!found) * -1;
}

template <class fn_type>
inline bool for_each_mask_seq_id(const uint64_t * mask,
                                 const int32_t words,
                                 const fn_type & fn) noexcept {
  bool ok = true;
  int32_t word_limit = words;
  for (int32_t w = 0; w < word_limit; ++w) {
    uint64_t bits = mask[static_cast<size_t>(w)];
    while (bits != 0U) {
      const int32_t bit = static_cast<int32_t>(std::countr_zero(bits));
      const int32_t seq_id = w * 64 + bit;
      ok = ok && fn(seq_id);
      const uint64_t next_bits = bits & (bits - 1U);
      const uint64_t continue_mask = uint64_t{0} - static_cast<uint64_t>(ok);
      bits = next_bits & continue_mask;
    }
    const int32_t keep_limit = static_cast<int32_t>(ok);
    word_limit = keep_limit * word_limit + (1 - keep_limit) * (w + 1);
  }
  return ok;
}

inline bool primary_ids_in_range(const int32_t * primary_ids,
                                 const int32_t count,
                                 const int32_t seq_limit) noexcept {
  bool in_range = true;
  int32_t scan_limit = count;
  for (int32_t i = 0; i < scan_limit; ++i) {
    const int32_t seq_id = primary_ids[i];
    in_range = in_range && seq_id >= 0 && seq_id < seq_limit;
    const int32_t keep_limit = static_cast<int32_t>(in_range);
    scan_limit = keep_limit * scan_limit + (1 - keep_limit) * (i + 1);
  }
  return in_range;
}

inline bool masks_have_non_empty_rows(const event::batch & req) noexcept {
  const bool has_masks = has_seq_masks_input(req);
  const int32_t mask_words = req.seq_mask_words;
  bool non_empty_rows = true;
  for (int32_t i = 0; i < req.n_tokens && has_masks && non_empty_rows; ++i) {
    const uint64_t * in_mask = req.seq_masks + static_cast<size_t>(i) * mask_words;
    non_empty_rows = non_empty_rows && !mask_empty(in_mask, mask_words);
  }
  return !has_masks || non_empty_rows;
}

inline bool primary_in_mask_when_both_inputs(const event::batch & req) noexcept {
  const bool has_masks = has_seq_masks_input(req);
  const bool has_primary = has_seq_primary_input(req);
  const bool check_required = has_masks && has_primary;
  const int32_t mask_words = req.seq_mask_words;
  bool primary_present = true;
  for (int32_t i = 0; i < req.n_tokens && check_required && primary_present; ++i) {
    const int32_t primary = req.seq_primary_ids[i];
    const uint64_t * in_mask = req.seq_masks + static_cast<size_t>(i) * mask_words;
    primary_present = primary_present && mask_has_bit(in_mask, mask_words, primary);
  }
  return !check_required || primary_present;
}

inline void continuity_track_active_none(std::array<int32_t, action::MAX_SEQ> &,
                                         const int32_t,
                                         const int32_t) noexcept {}

inline void continuity_track_active_some(std::array<int32_t, action::MAX_SEQ> & active_seq_ids,
                                         const int32_t active_seq_count,
                                         const int32_t seq_id) noexcept {
  active_seq_ids[static_cast<size_t>(active_seq_count)] = seq_id;
}

inline void continuity_track_active(std::array<int32_t, action::MAX_SEQ> & active_seq_ids,
                                    const int32_t active_seq_count,
                                    const int32_t seq_id,
                                    const bool track_active) noexcept {
  constexpr std::array<void (*)(std::array<int32_t, action::MAX_SEQ> &, int32_t, int32_t), 2>
      handlers = {
          continuity_track_active_none,
          continuity_track_active_some,
      };
  handlers[static_cast<size_t>(track_active)](active_seq_ids, active_seq_count, seq_id);
}

inline bool single_output_per_seq_ok(const event::batch_runtime & ev) noexcept {
  const auto & req = ev.request;
  const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
  const uint64_t * seq_masks_out = seq_masks_out_ptr(req);
  const int8_t * output_mask_out = output_mask_out_ptr(req);
  std::array<int32_t, action::MAX_SEQ> seq_output_count = {};

  bool ok = true;
  for (int32_t i = 0; i < req.n_tokens && ok; ++i) {
    const bool active = output_mask_out[i] != 0;
    const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
    const bool row_ok = !active || for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
                          seq_output_count[seq_id] += 1;
                          return seq_output_count[seq_id] <= 1;
                        });
    ok = ok && row_ok;
  }

  return ok;
}

inline bool continuity_ok(const event::batch_runtime & ev) noexcept {
  const auto & req = ev.request;
  const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
  const uint64_t * seq_masks_out = seq_masks_out_ptr(req);
  const int32_t * positions_out = positions_out_ptr(req);

  std::array<int32_t, action::MAX_SEQ> seq_last_pos = {};
  std::array<int32_t, action::MAX_SEQ> seq_pos_min = {};
  std::array<int32_t, action::MAX_SEQ> seq_pos_max = {};
  std::array<int32_t, action::MAX_SEQ> seq_pos_count = {};
  std::array<uint8_t, action::MAX_SEQ> seq_seen = {};
  std::array<int32_t, action::MAX_SEQ> active_seq_ids = {};
  std::array<uint64_t, static_cast<size_t>(action::MAX_SEQ * action::SEQ_WORDS)> cur_seq_set = {};
  int32_t active_seq_count = 0;

  seq_last_pos.fill(-1);
  seq_pos_min.fill(std::numeric_limits<int32_t>::max());
  seq_pos_max.fill(std::numeric_limits<int32_t>::min());

  bool ok = true;
  for (int32_t i = 0; i < req.n_tokens && ok; ++i) {
    const int32_t pos = positions_out[i];
    const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;

    ok = ok && for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
           const int32_t last = seq_last_pos[seq_id];
           const bool monotonic = (last < 0) || pos >= last;

           const bool pos_changed = pos != last;
           seq_pos_count[seq_id] += static_cast<int32_t>(pos_changed);
           seq_last_pos[seq_id] = pos;
           seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], pos);
           seq_pos_max[seq_id] = std::max(seq_pos_max[seq_id], pos);

           uint64_t * cur_mask = cur_seq_set.data() + static_cast<size_t>(seq_id * mask_words);
           const bool first_seen = seq_seen[seq_id] == 0U;
           const bool has_active_slot = active_seq_count < action::MAX_SEQ;
           const bool track_active = first_seen && has_active_slot;
           continuity_track_active(active_seq_ids, active_seq_count, seq_id, track_active);
           active_seq_count += static_cast<int32_t>(track_active);
           seq_seen[seq_id] =
               static_cast<uint8_t>(seq_seen[seq_id] | static_cast<uint8_t>(first_seen));

           const uint64_t first_seen_mask = uint64_t{0} - static_cast<uint64_t>(first_seen);
           for (int32_t mw = 0; mw < mask_words; ++mw) {
             cur_mask[static_cast<size_t>(mw)] =
                 (~uint64_t{0} & first_seen_mask) |
                 (cur_mask[static_cast<size_t>(mw)] & ~first_seen_mask);
             cur_mask[static_cast<size_t>(mw)] &= mask[static_cast<size_t>(mw)];
           }

           return (!first_seen || has_active_slot) && monotonic && !mask_empty(cur_mask, mask_words);
         });
  }

  for (int32_t i = 0; i < active_seq_count && ok; ++i) {
    const int32_t seq_id = active_seq_ids[i];
    const int32_t min_pos = seq_pos_min[seq_id];
    const int32_t max_pos = seq_pos_max[seq_id];
    const int32_t count = seq_pos_count[seq_id];
    const bool has_bounds = min_pos != std::numeric_limits<int32_t>::max() &&
                            max_pos != std::numeric_limits<int32_t>::min();
    const int64_t span = static_cast<int64_t>(max_pos) - static_cast<int64_t>(min_pos) + 1;
    ok = ok && (!has_bounds || span <= count);
  }

  return ok;
}

inline probe_status seeded_generation_probe(
    const event::batch_runtime & ev,
    std::array<int32_t, action::MAX_SEQ> & seeded_next_pos_out) noexcept {
  constexpr std::array<std::array<probe_status, 2>, 2> status_lut = {{
      {probe_status::backend_error, probe_status::backend_error},
      {probe_status::invalid, probe_status::ok},
  }};

  const auto & req = ev.request;
  const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
  const int32_t * seq_primary_ids_out = seq_primary_ids_out_ptr(req);
  const uint64_t * seq_masks_out = seq_masks_out_ptr(req);
  std::array<int32_t, action::MAX_SEQ> next_pos = {};

  bool backend_ok = true;
  bool valid = true;
  for (int32_t seq_id = 0; seq_id < action::MAX_SEQ && backend_ok && valid; ++seq_id) {
    int32_t seed = 0;
    const bool resolved = req.resolve_position_seed(req.position_seed_ctx, seq_id, &seed);
    backend_ok = backend_ok && resolved;
    valid = valid && seed >= 0;
    next_pos[seq_id] = seed;
  }
  seeded_next_pos_out = next_pos;

  for (int32_t i = 0; i < req.n_tokens && backend_ok && valid; ++i) {
    const int32_t primary = seq_primary_ids_out[i];
    const int32_t pos = next_pos[primary];
    valid = valid && pos != std::numeric_limits<int32_t>::max();

    const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
    const bool compatible =
        valid &&
        for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
          return next_pos[seq_id] == pos;
        });
    valid = valid && compatible;
    while (valid) {
      for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
        next_pos[seq_id] = pos + 1;
        return true;
      });
      break;
    }
  }

  return status_lut[static_cast<size_t>(backend_ok)][static_cast<size_t>(valid)];
}

inline bool unseeded_generation_probe(const event::batch_runtime & ev) noexcept {
  const auto & req = ev.request;
  const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
  const int32_t * seq_primary_ids_out = seq_primary_ids_out_ptr(req);
  const uint64_t * seq_masks_out = seq_masks_out_ptr(req);
  std::array<int32_t, action::MAX_SEQ> next_pos = {};
  std::array<uint8_t, action::MAX_SEQ> seeded = {};

  bool valid = true;
  for (int32_t i = 0; i < req.n_tokens && valid; ++i) {
    const int32_t primary = seq_primary_ids_out[i];
    const int32_t pos = next_pos[primary];
    valid = valid && pos != std::numeric_limits<int32_t>::max();

    const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
    const bool aligned = valid && for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
                           const bool first_seen = seeded[seq_id] == 0U;
                           const int32_t current = static_cast<int32_t>(!first_seen) * next_pos[seq_id] +
                                                   static_cast<int32_t>(first_seen) * pos;
                           return current == pos;
                         });
    valid = valid && aligned;
    while (valid) {
      for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
        seeded[seq_id] = 1U;
        next_pos[seq_id] = pos + 1;
        return true;
      });
      break;
    }
  }

  return valid;
}

}  // namespace emel::token::batcher::detail
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
    const size_t is_ok = static_cast<size_t>(status == detail::probe_status::ok);
    const size_t is_backend =
        static_cast<size_t>(status == detail::probe_status::backend_error);
    constexpr std::array<position_probe_status, 3> mapped_status = {
        position_probe_status::invalid,
        position_probe_status::ok,
        position_probe_status::backend_error,
    };
    ctx.seeded_probe_status = mapped_status[is_ok + (is_backend << 1u)];
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
