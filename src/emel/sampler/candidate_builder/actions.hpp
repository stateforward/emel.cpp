#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/candidate_builder/events.hpp"

namespace emel::sampler::candidate_builder::action {

struct context {
  int32_t candidate_count = 0;
};

inline constexpr auto begin_build = [](const event::build & ev, context & ctx) {
  ctx.candidate_count = 0;
  (void)ev;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (
      ev.logits == nullptr ||
      ev.vocab_size <= 0 ||
      ev.candidate_ids_out == nullptr ||
      ev.candidate_scores_out == nullptr ||
      ev.candidate_capacity < ev.vocab_size ||
      ev.candidate_count_out == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
  (void)ctx;
};

inline constexpr auto run_build_candidates = [](const event::build_candidates & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ev.vocab_size <= 0 || ev.logits == nullptr || ev.candidate_ids_out == nullptr ||
      ev.candidate_scores_out == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  for (int32_t i = 0; i < ev.vocab_size; ++i) {
    ev.candidate_ids_out[i] = i;
    ev.candidate_scores_out[i] = ev.logits[i];
  }
  ctx.candidate_count = ev.vocab_size;
};

inline constexpr auto run_normalize_scores = [](const event::normalize_scores & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.candidate_count <= 0) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }
  if (ev.candidate_scores_out == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  float max_score = ev.candidate_scores_out[0];
  for (int32_t i = 1; i < ctx.candidate_count; ++i) {
    if (ev.candidate_scores_out[i] > max_score) {
      max_score = ev.candidate_scores_out[i];
    }
  }

  for (int32_t i = 0; i < ctx.candidate_count; ++i) {
    ev.candidate_scores_out[i] -= max_score;
  }
};

inline constexpr auto on_build_done = [](const events::build_done & ev, context & ctx) {
  if (ev.candidate_count_out != nullptr) {
    *ev.candidate_count_out = ev.candidate_count;
  }
  (void)ctx;
};

inline constexpr auto on_build_error = [](const events::build_error & ev, context & ctx) {
  if (ev.candidate_count_out != nullptr) {
    *ev.candidate_count_out = 0;
  }
  (void)ctx;
};

}  // namespace emel::sampler::candidate_builder::action
