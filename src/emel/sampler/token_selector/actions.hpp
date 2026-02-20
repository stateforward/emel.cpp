#pragma once

#include <cmath>

#include "emel/sampler/token_selector/context.hpp"

namespace emel::sampler::token_selector::action {

inline void clear_request(context & ctx) noexcept {
  ctx.request = nullptr;
}

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

struct begin_select_token {
  void operator()(const event::select_token & ev, context & ctx) const noexcept {
    ctx.request = &ev;
    ctx.candidate_count = ev.candidate_count;
    ctx.policy = ev.policy;
    ctx.random_01 = ev.random_01;
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

struct run_validate {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::select_token * request = ctx.request;
    if (request == nullptr ||
        request->candidate_ids == nullptr ||
        request->candidate_scores == nullptr ||
        request->candidate_count <= 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (request->policy == event::selection_policy::categorical &&
        (request->random_01 < 0.0f || request->random_01 >= 1.0f)) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct run_select {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::select_token * request = ctx.request;
    if (request == nullptr ||
        request->candidate_ids == nullptr ||
        request->candidate_scores == nullptr ||
        request->candidate_count <= 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    if (request->policy == event::selection_policy::argmax) {
      int32_t best_index = 0;
      float best_score = request->candidate_scores[0];
      for (int32_t i = 1; i < request->candidate_count; ++i) {
        if (request->candidate_scores[i] > best_score) {
          best_score = request->candidate_scores[i];
          best_index = i;
        }
      }
      ctx.selected_token = request->candidate_ids[best_index];
      if (request->selected_token_out != nullptr) {
        *request->selected_token_out = ctx.selected_token;
      }
      return;
    }

    float max_score = request->candidate_scores[0];
    for (int32_t i = 1; i < request->candidate_count; ++i) {
      if (request->candidate_scores[i] > max_score) {
        max_score = request->candidate_scores[i];
      }
    }

    float total_weight = 0.0f;
    for (int32_t i = 0; i < request->candidate_count; ++i) {
      total_weight += std::exp(request->candidate_scores[i] - max_score);
    }

    if (!(total_weight > 0.0f)) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }

    const float target = request->random_01 * total_weight;
    float cumulative = 0.0f;
    for (int32_t i = 0; i < request->candidate_count; ++i) {
      cumulative += std::exp(request->candidate_scores[i] - max_score);
      if (cumulative >= target) {
        ctx.selected_token = request->candidate_ids[i];
        if (request->selected_token_out != nullptr) {
          *request->selected_token_out = ctx.selected_token;
        }
        return;
      }
    }

    ctx.selected_token = request->candidate_ids[request->candidate_count - 1];
    if (request->selected_token_out != nullptr) {
      *request->selected_token_out = ctx.selected_token;
    }
  }
};

struct publish_done {
  void operator()(context & ctx) const noexcept {
    const event::select_token * request = ctx.request;
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
    const event::select_token * request = ctx.request;
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

inline constexpr begin_select_token begin_select_token{};
inline constexpr set_invalid_argument set_invalid_argument{};
inline constexpr set_backend_error set_backend_error{};
inline constexpr run_validate run_validate{};
inline constexpr run_select run_select{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::sampler::token_selector::action
