#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/pipeline/events.hpp"

namespace emel::sampler::pipeline::action {

struct context {
  const event::sample * request = nullptr;
  int32_t candidate_count = 0;
  int32_t sampler_count = 0;
  int32_t sampler_index = 0;
  int32_t selected_token = -1;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

inline void clear_request(context & ctx) noexcept {
  ctx.request = nullptr;
}

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

struct begin_sample {
  void operator()(const event::sample & ev, context & ctx) const noexcept {
    ctx.request = &ev;
    ctx.candidate_count = 0;
    ctx.sampler_count = ev.sampler_count;
    ctx.sampler_index = 0;
    ctx.selected_token = -1;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct set_invalid_argument {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_INVALID_ARGUMENT); }
};

struct set_backend_error {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

struct run_prepare_candidates {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::sample * request = ctx.request;
    if (request == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (request->logits == nullptr ||
        request->vocab_size <= 0 ||
        request->candidate_ids == nullptr ||
        request->candidate_scores == nullptr ||
        request->candidate_capacity < request->vocab_size ||
        ctx.sampler_count < 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    for (int32_t i = 0; i < request->vocab_size; ++i) {
      request->candidate_ids[i] = i;
      request->candidate_scores[i] = request->logits[i];
    }
    ctx.candidate_count = request->vocab_size;
  }
};

struct run_apply_sampling {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::sample * request = ctx.request;
    if (request == nullptr ||
        request->sampler_fns == nullptr ||
        ctx.candidate_count <= 0 ||
        ctx.sampler_index < 0 ||
        ctx.sampler_index >= ctx.sampler_count) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    event::sampler_fn fn = request->sampler_fns[ctx.sampler_index];
    if (fn == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t sampler_error = EMEL_OK;
    const bool ok = fn(
      request->candidate_ids,
      request->candidate_scores,
      ctx.candidate_count,
      request->sampler_user_data,
      &sampler_error);
    if (!ok || sampler_error != EMEL_OK) {
      set_error(ctx, sampler_error == EMEL_OK ? EMEL_ERR_BACKEND : sampler_error);
      return;
    }
    ctx.sampler_index += 1;
  }
};

struct run_select_token {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::sample * request = ctx.request;
    if (request == nullptr ||
        ctx.candidate_count <= 0 ||
        request->candidate_ids == nullptr ||
        request->candidate_scores == nullptr) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }
    int32_t best_id = request->candidate_ids[0];
    float best_score = request->candidate_scores[0];
    for (int32_t i = 1; i < ctx.candidate_count; ++i) {
      if (request->candidate_scores[i] > best_score) {
        best_score = request->candidate_scores[i];
        best_id = request->candidate_ids[i];
      }
    }
    ctx.selected_token = best_id;
    if (request->selected_token_out != nullptr) {
      *request->selected_token_out = best_id;
    }
  }
};

struct publish_done {
  void operator()(context & ctx) const noexcept {
    const event::sample * request = ctx.request;
    if (request == nullptr) {
      return;
    }
    if (request->error_out != nullptr) {
      *request->error_out = EMEL_OK;
    }
    clear_request(ctx);
  }
};

struct publish_error {
  void operator()(context & ctx) const noexcept {
    const event::sample * request = ctx.request;
    if (request == nullptr) {
      return;
    }
    int32_t err = ctx.last_error;
    if (err == EMEL_OK) {
      err = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
    }
    ctx.last_error = err;
    if (request->error_out != nullptr) {
      *request->error_out = err;
    }
    clear_request(ctx);
  }
};

struct on_unexpected {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

inline constexpr begin_sample begin_sample{};
inline constexpr set_invalid_argument set_invalid_argument{};
inline constexpr set_backend_error set_backend_error{};
inline constexpr run_prepare_candidates run_prepare_candidates{};
inline constexpr run_apply_sampling run_apply_sampling{};
inline constexpr run_select_token run_select_token{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::sampler::pipeline::action
