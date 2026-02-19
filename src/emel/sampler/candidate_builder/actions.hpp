#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/candidate_builder/events.hpp"

namespace emel::sampler::candidate_builder::action {

struct context {
  const event::build * request = nullptr;
  int32_t candidate_count = 0;
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

struct begin_build {
  void operator()(const event::build & ev, context & ctx) const noexcept {
    ctx.request = &ev;
    ctx.candidate_count = 0;
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
    const event::build * request = ctx.request;
    if (request == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (request->logits == nullptr ||
        request->vocab_size <= 0 ||
        request->candidate_ids_out == nullptr ||
        request->candidate_scores_out == nullptr ||
        request->candidate_capacity < request->vocab_size ||
        request->candidate_count_out == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    }
  }
};

struct run_build_candidates {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::build * request = ctx.request;
    if (request == nullptr ||
        request->logits == nullptr ||
        request->vocab_size <= 0 ||
        request->candidate_ids_out == nullptr ||
        request->candidate_scores_out == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    for (int32_t i = 0; i < request->vocab_size; ++i) {
      request->candidate_ids_out[i] = i;
      request->candidate_scores_out[i] = request->logits[i];
    }
    ctx.candidate_count = request->vocab_size;
  }
};

struct run_normalize_scores {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::build * request = ctx.request;
    if (request == nullptr || request->candidate_scores_out == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    if (ctx.candidate_count <= 0) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }
    float max_score = request->candidate_scores_out[0];
    for (int32_t i = 1; i < ctx.candidate_count; ++i) {
      if (request->candidate_scores_out[i] > max_score) {
        max_score = request->candidate_scores_out[i];
      }
    }
    for (int32_t i = 0; i < ctx.candidate_count; ++i) {
      request->candidate_scores_out[i] -= max_score;
    }
  }
};

struct publish_done {
  void operator()(context & ctx) const noexcept {
    const event::build * request = ctx.request;
    if (request == nullptr) {
      return;
    }
    if (request->error_out != nullptr) {
      *request->error_out = EMEL_OK;
    }
    if (request->candidate_count_out != nullptr) {
      *request->candidate_count_out = ctx.candidate_count;
    }
    clear_request(ctx);
  }
};

struct publish_error {
  void operator()(context & ctx) const noexcept {
    const event::build * request = ctx.request;
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
    if (request->candidate_count_out != nullptr) {
      *request->candidate_count_out = 0;
    }
    clear_request(ctx);
  }
};

struct on_unexpected {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

inline constexpr begin_build begin_build{};
inline constexpr set_invalid_argument set_invalid_argument{};
inline constexpr set_backend_error set_backend_error{};
inline constexpr run_validate run_validate{};
inline constexpr run_build_candidates run_build_candidates{};
inline constexpr run_normalize_scores run_normalize_scores{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::sampler::candidate_builder::action
