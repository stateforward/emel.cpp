#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/emel.h"
#include "emel/text/conditioner/context.hpp"
#include "emel/text/conditioner/events.hpp"

namespace emel::text::conditioner::action {

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

inline void clear_prepare_request(context & ctx) noexcept {
  ctx.input = {};
  ctx.formatted_length = 0;
  ctx.add_special = true;
  ctx.parse_special = false;
  ctx.token_ids_out = nullptr;
  ctx.token_capacity = 0;
  ctx.token_count = 0;
}

struct begin_bind {
  void operator()(const event::bind & ev, context & ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.vocab = ev.vocab;
    ctx.preprocessor_variant = ev.preprocessor_variant;
    ctx.encoder_variant = ev.encoder_variant;
    ctx.tokenizer_sm = ev.tokenizer_sm;
    ctx.dispatch_tokenizer_bind = ev.dispatch_tokenizer_bind;
    ctx.dispatch_tokenizer_tokenize = ev.dispatch_tokenizer_tokenize;
    ctx.formatter_ctx = ev.formatter_ctx;
    ctx.format_prompt = ev.format_prompt;
    ctx.add_special_default = ev.add_special;
    ctx.parse_special_default = ev.parse_special;
    ctx.is_bound = false;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    clear_prepare_request(ctx);
  }
};

struct reject_bind {
  void operator()(const event::bind &, context & ctx) const noexcept {
    ctx.is_bound = false;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct bind_tokenizer {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.is_bound = false;
    if (ctx.vocab == nullptr || ctx.tokenizer_sm == nullptr ||
        ctx.dispatch_tokenizer_bind == nullptr ||
        ctx.dispatch_tokenizer_tokenize == nullptr ||
        ctx.format_prompt == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    int32_t err = EMEL_OK;
    emel::text::tokenizer::event::bind tokenize_bind = {};
    tokenize_bind.vocab = ctx.vocab;
    tokenize_bind.preprocessor_variant = ctx.preprocessor_variant;
    tokenize_bind.encoder_variant = ctx.encoder_variant;
    tokenize_bind.error_out = &err;
    const bool accepted =
        ctx.dispatch_tokenizer_bind(ctx.tokenizer_sm, tokenize_bind);
    if (!accepted && err == EMEL_OK) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }
    if (err != EMEL_OK) {
      set_error(ctx, err);
      return;
    }

    ctx.is_bound = true;
    ctx.last_error = EMEL_OK;
  }
};

struct begin_prepare {
  void operator()(const event::prepare & ev, context & ctx) const noexcept {
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    ctx.input = ev.input;
    ctx.formatted_length = 0;
    ctx.add_special =
        ev.use_bind_defaults ? ctx.add_special_default : ev.add_special;
    ctx.parse_special =
        ev.use_bind_defaults ? ctx.parse_special_default : ev.parse_special;
    ctx.token_ids_out = ev.token_ids_out;
    ctx.token_capacity = ev.token_capacity;
    ctx.token_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct reject_prepare {
  void operator()(const event::prepare &, context & ctx) const noexcept {
    ctx.token_count = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct run_format {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;

    emel::text::formatter::format_request request = {};
    request.input = ctx.input;
    request.output = ctx.formatted.data();
    request.output_capacity = ctx.formatted.size();
    request.output_length_out = &ctx.formatted_length;

    int32_t err = EMEL_OK;
    const bool accepted =
        ctx.format_prompt(ctx.formatter_ctx, request, &err);
    if (!accepted && err == EMEL_OK) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }
    if (err != EMEL_OK) {
      set_error(ctx, err);
      return;
    }
    if (ctx.formatted_length > ctx.formatted.size()) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
  }
};

struct run_tokenize {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    if (ctx.dispatch_tokenizer_tokenize == nullptr || ctx.tokenizer_sm == nullptr ||
        ctx.vocab == nullptr || ctx.token_ids_out == nullptr ||
        ctx.token_capacity <= 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    int32_t err = EMEL_OK;
    int32_t count = 0;
    emel::text::tokenizer::event::tokenize tokenize_ev = {};
    tokenize_ev.vocab = ctx.vocab;
    tokenize_ev.text = std::string_view(ctx.formatted.data(), ctx.formatted_length);
    tokenize_ev.add_special = ctx.add_special;
    tokenize_ev.parse_special = ctx.parse_special;
    tokenize_ev.token_ids_out = ctx.token_ids_out;
    tokenize_ev.token_capacity = ctx.token_capacity;
    tokenize_ev.token_count_out = &count;
    tokenize_ev.error_out = &err;

    const bool accepted =
        ctx.dispatch_tokenizer_tokenize(ctx.tokenizer_sm, tokenize_ev);
    if (!accepted && err == EMEL_OK) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }
    if (err != EMEL_OK) {
      set_error(ctx, err);
      return;
    }
    if (count < 0 || count > ctx.token_capacity) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }

    ctx.token_count = count;
    ctx.last_error = EMEL_OK;
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
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
  template <class event_type>
  void operator()(const event_type &, context & ctx) const noexcept {
    ctx.token_count = 0;
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

inline constexpr begin_bind begin_bind{};
inline constexpr reject_bind reject_bind{};
inline constexpr bind_tokenizer bind_tokenizer{};
inline constexpr begin_prepare begin_prepare{};
inline constexpr reject_prepare reject_prepare{};
inline constexpr run_format run_format{};
inline constexpr run_tokenize run_tokenize{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::conditioner::action
