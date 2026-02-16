#pragma once

#include <cmath>
#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/token_selector/events.hpp"

namespace emel::sampler::token_selector::action {

struct context {
  int32_t candidate_count = 0;
  event::selection_policy policy = event::selection_policy::argmax;
  float random_01 = 0.0f;
  int32_t selected_token = -1;
};

inline constexpr auto begin_select_token = [](const event::select_token & ev, context & ctx) {
  ctx.candidate_count = ev.candidate_count;
  ctx.policy = ev.policy;
  ctx.random_01 = ev.random_01;
  ctx.selected_token = -1;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (
      ev.candidate_ids == nullptr ||
      ev.candidate_scores == nullptr ||
      ev.candidate_count <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (
      ev.policy == event::selection_policy::categorical &&
      (ev.random_01 < 0.0f || ev.random_01 >= 1.0f)) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
  }
  (void)ctx;
};

inline constexpr auto run_select = [](const event::select & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ev.candidate_count <= 0 || ev.candidate_ids == nullptr || ev.candidate_scores == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (ev.policy == event::selection_policy::argmax) {
    int32_t best_index = 0;
    float best_score = ev.candidate_scores[0];
    for (int32_t i = 1; i < ev.candidate_count; ++i) {
      if (ev.candidate_scores[i] > best_score) {
        best_score = ev.candidate_scores[i];
        best_index = i;
      }
    }
    ctx.selected_token = ev.candidate_ids[best_index];
    if (ev.selected_token_out != nullptr) {
      *ev.selected_token_out = ctx.selected_token;
    }
    return;
  }

  float max_score = ev.candidate_scores[0];
  for (int32_t i = 1; i < ev.candidate_count; ++i) {
    if (ev.candidate_scores[i] > max_score) {
      max_score = ev.candidate_scores[i];
    }
  }

  float total_weight = 0.0f;
  for (int32_t i = 0; i < ev.candidate_count; ++i) {
    total_weight += std::exp(ev.candidate_scores[i] - max_score);
  }

  if (!(total_weight > 0.0f)) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }

  const float target = ev.random_01 * total_weight;
  float cumulative = 0.0f;
  for (int32_t i = 0; i < ev.candidate_count; ++i) {
    cumulative += std::exp(ev.candidate_scores[i] - max_score);
    if (cumulative >= target) {
      ctx.selected_token = ev.candidate_ids[i];
      if (ev.selected_token_out != nullptr) {
        *ev.selected_token_out = ctx.selected_token;
      }
      return;
    }
  }

  ctx.selected_token = ev.candidate_ids[ev.candidate_count - 1];
  if (ev.selected_token_out != nullptr) {
    *ev.selected_token_out = ctx.selected_token;
  }
};

inline constexpr auto on_token_selection_done = [](const events::token_selection_done & ev, context & ctx) {
  ctx.selected_token = ev.token_id;
};

inline constexpr auto on_token_selection_error = [](const events::token_selection_error & ev, context & ctx) {
  (void)ev;
  (void)ctx;
};

}  // namespace emel::sampler::token_selector::action
