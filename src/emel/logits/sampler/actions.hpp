#pragma once

#include <cmath>

#include "emel/logits/sampler/context.hpp"

namespace emel::logits::sampler::action {

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

struct exec_prepare_candidates {
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

struct exec_apply_samplers {
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

struct exec_select_token {
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

    if (request->policy == event::selection_policy::categorical &&
        (request->random_01 < 0.0f || request->random_01 >= 1.0f)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    if (request->policy == event::selection_policy::argmax) {
      int32_t best_id = request->candidate_ids[0];
      float best_score = request->candidate_scores[0];
      for (int32_t i = 1; i < ctx.candidate_count; ++i) {
        const float score = request->candidate_scores[i];
        const int32_t token_id = request->candidate_ids[i];
        if (score > best_score || (score == best_score && token_id < best_id)) {
          best_score = score;
          best_id = token_id;
        }
      }
      ctx.selected_token = best_id;
      if (request->selected_token_out != nullptr) {
        *request->selected_token_out = best_id;
      }
      return;
    }

    float max_score = request->candidate_scores[0];
    for (int32_t i = 1; i < ctx.candidate_count; ++i) {
      if (request->candidate_scores[i] > max_score) {
        max_score = request->candidate_scores[i];
      }
    }

    float total_weight = 0.0f;
    for (int32_t i = 0; i < ctx.candidate_count; ++i) {
      total_weight += std::exp(request->candidate_scores[i] - max_score);
    }
    if (!(total_weight > 0.0f)) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }

    const float target = request->random_01 * total_weight;
    float cumulative = 0.0f;
    for (int32_t i = 0; i < ctx.candidate_count; ++i) {
      cumulative += std::exp(request->candidate_scores[i] - max_score);
      if (cumulative >= target) {
        ctx.selected_token = request->candidate_ids[i];
        if (request->selected_token_out != nullptr) {
          *request->selected_token_out = ctx.selected_token;
        }
        return;
      }
    }

    ctx.selected_token = request->candidate_ids[ctx.candidate_count - 1];
    if (request->selected_token_out != nullptr) {
      *request->selected_token_out = ctx.selected_token;
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
inline constexpr exec_prepare_candidates exec_prepare_candidates{};
inline constexpr exec_apply_samplers exec_apply_samplers{};
inline constexpr exec_select_token exec_select_token{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::logits::sampler::action
