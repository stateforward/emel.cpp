#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/pipeline/events.hpp"

namespace emel::sampler::pipeline::action {

struct context {
  int32_t candidate_count = 0;

  int32_t sampler_count = 0;
  int32_t sampler_index = 0;

  int32_t selected_token = -1;
};

inline constexpr auto begin_sample = [](const event::sample & ev, context & ctx) {
  ctx.candidate_count = 0;

  ctx.sampler_count = ev.sampler_count;
  ctx.sampler_index = 0;
  ctx.selected_token = -1;
  (void)ev;
};

inline constexpr auto run_prepare_candidates = [](const event::prepare_candidates & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (
      ev.logits == nullptr ||
      ev.vocab_size <= 0 ||
      ev.candidate_ids == nullptr ||
      ev.candidate_scores == nullptr ||
      ev.candidate_capacity < ev.vocab_size ||
      ctx.sampler_count < 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  for (int32_t i = 0; i < ev.vocab_size; ++i) {
    ev.candidate_ids[i] = i;
    ev.candidate_scores[i] = ev.logits[i];
  }
  ctx.candidate_count = ev.vocab_size;
};

inline constexpr auto run_apply_sampling = [](const event::apply_sampling & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.sampler_index >= ctx.sampler_count) {
    return;
  }

  if (ev.sampler_fns == nullptr || ctx.candidate_count <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  const event::sampler_fn fn = ev.sampler_fns[ctx.sampler_index];
  if (fn == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  int32_t sampler_error = EMEL_OK;
  const bool ok = fn(
      ev.candidate_ids,
      ev.candidate_scores,
      ctx.candidate_count,
      ev.sampler_user_data,
      &sampler_error);
  if (!ok || sampler_error != EMEL_OK) {
    *ev.error_out = sampler_error == EMEL_OK ? EMEL_ERR_BACKEND : sampler_error;
    return;
  }

  ctx.sampler_index += 1;
};

inline constexpr auto run_select_token = [](const event::select_token & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.candidate_count <= 0 || ev.candidate_ids == nullptr || ev.candidate_scores == nullptr) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }

  int32_t best_id = ev.candidate_ids[0];
  float best_score = ev.candidate_scores[0];
  for (int32_t i = 1; i < ctx.candidate_count; ++i) {
    if (ev.candidate_scores[i] > best_score) {
      best_score = ev.candidate_scores[i];
      best_id = ev.candidate_ids[i];
    }
  }

  ctx.selected_token = best_id;
  if (ev.selected_token_out != nullptr) {
    *ev.selected_token_out = best_id;
  }
};

inline constexpr auto on_sampling_done = [](const events::sampling_done &, context & ctx) {
  (void)ctx;
};

inline constexpr auto on_sampling_error = [](const events::sampling_error & ev, context & ctx) {
  (void)ev;
  (void)ctx;
};

}  // namespace emel::sampler::pipeline::action
