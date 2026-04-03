#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/text/conditioner/context.hpp"
#include "emel/text/conditioner/detail.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/conditioner/events.hpp"

namespace emel::text::conditioner::action {

template <class runtime_event_type>
inline void set_error(const runtime_event_type &runtime_ev, context &,
                      const error err) noexcept {
  const auto &ev = detail::unwrap_runtime_event(runtime_ev);
  ev.ctx.err = err;
  ev.ctx.result = false;
}

struct begin_bind {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);

    ctx.vocab = &ev.request.vocab;
    ctx.preprocessor_variant = ev.request.preprocessor_variant;
    ctx.encoder_variant = ev.request.encoder_variant;
    ctx.tokenizer_sm = ev.request.tokenizer_sm;
    ctx.dispatch_tokenizer_bind = ev.request.dispatch_tokenizer_bind;
    ctx.dispatch_tokenizer_tokenize = ev.request.dispatch_tokenizer_tokenize;
    ctx.formatter_ctx = ev.request.formatter_ctx;
    ctx.format_prompt = ev.request.format_prompt;
    ctx.formatter_contract = ev.request.formatter_contract;
    ctx.add_special_default = ev.request.add_special;
    ctx.parse_special_default = ev.request.parse_special;
    ctx.is_bound = false;

    ev.ctx.err = error::none;
    ev.ctx.result = false;
    ev.ctx.bind_accepted = false;
    ev.ctx.bind_err_code = detail::to_local_error_code(error::none);
  }
};

struct reject_bind {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    ctx.is_bound = false;
    set_error(runtime_ev, ctx, error::invalid_argument);
  }
};

struct dispatch_bind_tokenizer {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);

    int32_t err = detail::to_local_error_code(error::none);
    emel::text::tokenizer::event::bind tokenizer_bind = {};
    tokenizer_bind.vocab = ctx.vocab;
    tokenizer_bind.preprocessor_variant = ctx.preprocessor_variant;
    tokenizer_bind.encoder_variant = ctx.encoder_variant;
    tokenizer_bind.error_out = &err;

    ev.ctx.bind_accepted =
        ctx.dispatch_tokenizer_bind(ctx.tokenizer_sm, tokenizer_bind);
    ev.ctx.bind_err_code = err;
  }
};

struct bind_error_backend {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::backend);
  }
};

struct bind_success {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ctx.is_bound = true;
    ev.ctx.err = error::none;
    ev.ctx.result = true;
  }
};

struct begin_prepare_bind_defaults {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);

    ev.ctx.err = error::none;
    ev.ctx.formatted_length = 0;
    ev.ctx.add_special = ctx.add_special_default;
    ev.ctx.parse_special = ctx.parse_special_default;
    ev.ctx.token_count = 0;
    ev.ctx.result = false;
    ev.ctx.format_accepted = false;
    ev.ctx.format_err_code = detail::to_local_error_code(error::none);
    ev.ctx.tokenize_accepted = false;
    ev.ctx.tokenize_err_code = detail::to_local_error_code(error::none);
  }
};

struct begin_prepare_from_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);

    ev.ctx.err = error::none;
    ev.ctx.formatted_length = 0;
    ev.ctx.add_special = ev.request.add_special;
    ev.ctx.parse_special = ev.request.parse_special;
    ev.ctx.token_count = 0;
    ev.ctx.result = false;
    ev.ctx.format_accepted = false;
    ev.ctx.format_err_code = detail::to_local_error_code(error::none);
    ev.ctx.tokenize_accepted = false;
    ev.ctx.tokenize_err_code = detail::to_local_error_code(error::none);
  }
};

struct reject_prepare {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.token_count = 0;
    set_error(ev, ctx, error::invalid_argument);
  }
};

struct dispatch_format {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);

    emel::text::formatter::format_request request = {};
    request.messages = ev.request.messages;
    request.add_generation_prompt = ev.request.add_generation_prompt;
    request.enable_thinking = ev.request.enable_thinking;
    request.output = ev.ctx.formatted;
    request.output_capacity = ev.ctx.formatted_capacity;
    request.output_length_out = &ev.ctx.formatted_length;

    int32_t err = detail::to_local_error_code(error::none);
    ev.ctx.format_accepted =
        ctx.format_prompt(ctx.formatter_ctx, request, &err);
    ev.ctx.format_err_code = err;
  }
};

struct format_error_backend {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::backend);
  }
};

struct format_error_invalid_argument {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::invalid_argument);
  }
};

struct dispatch_tokenize {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);

    int32_t err = detail::to_local_error_code(error::none);
    int32_t count = 0;
    emel::text::tokenizer::event::tokenize tokenize_ev = {};
    tokenize_ev.vocab = ctx.vocab;
    tokenize_ev.text =
        std::string_view(ev.ctx.formatted, ev.ctx.formatted_length);
    tokenize_ev.add_special = ev.ctx.add_special;
    tokenize_ev.parse_special = ev.ctx.parse_special;
    tokenize_ev.token_ids_out = ev.request.token_ids_out;
    tokenize_ev.token_capacity = ev.request.token_capacity;
    tokenize_ev.token_count_out = &count;
    tokenize_ev.error_out = &err;

    ev.ctx.tokenize_accepted =
        ctx.dispatch_tokenizer_tokenize(ctx.tokenizer_sm, tokenize_ev);
    ev.ctx.tokenize_err_code = err;
    ev.ctx.token_count = count;
  }
};

struct set_error_invalid_argument {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::invalid_argument);
  }
};

struct set_error_model_invalid {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::model_invalid);
  }
};

struct set_error_capacity {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::capacity);
  }
};

struct set_error_backend {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::backend);
  }
};

struct set_error_untracked {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::untracked);
  }
};

struct tokenize_error_backend {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    set_error(runtime_ev, ctx, error::backend);
  }
};

struct prepare_success {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = error::none;
    ev.ctx.result = true;
  }
};

struct write_bind_error_out {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    *ev.request.error_out = detail::to_local_error_code(ev.ctx.err);
  }
};

struct emit_bind_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.request.dispatch_done(ev.request.owner_sm,
                             events::binding_done{&ev.request});
  }
};

struct emit_bind_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.request.dispatch_error(ev.request.owner_sm,
                              events::binding_error{
                                  &ev.request,
                                  detail::to_local_error_code(ev.ctx.err),
                              });
  }
};

struct write_prepare_token_count {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.request.token_count_out = ev.ctx.token_count;
  }
};

struct write_prepare_error_out {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.request.error_out = detail::to_local_error_code(ev.ctx.err);
  }
};

struct emit_prepare_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.request.dispatch_done(ev.request.owner_sm, events::conditioning_done{
                                                      &ev.request,
                                                      ev.ctx.token_count,
                                                  });
  }
};

struct emit_prepare_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.request.dispatch_error(ev.request.owner_sm,
                              events::conditioning_error{
                                  &ev.request,
                                  detail::to_local_error_code(ev.ctx.err),
                              });
  }
};

struct on_unexpected {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    if constexpr (requires { ev.ctx.token_count; }) {
      ev.ctx.token_count = 0;
    }
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = error::invalid_argument;
    }
    if constexpr (requires { ev.ctx.result; }) {
      ev.ctx.result = false;
    }
  }
};

inline constexpr begin_bind begin_bind{};
inline constexpr reject_bind reject_bind{};
inline constexpr dispatch_bind_tokenizer dispatch_bind_tokenizer{};
inline constexpr bind_error_backend bind_error_backend{};
inline constexpr bind_success bind_success{};
inline constexpr begin_prepare_bind_defaults begin_prepare_bind_defaults{};
inline constexpr begin_prepare_from_request begin_prepare_from_request{};
inline constexpr reject_prepare reject_prepare{};
inline constexpr dispatch_format dispatch_format{};
inline constexpr format_error_backend format_error_backend{};
inline constexpr format_error_invalid_argument format_error_invalid_argument{};
inline constexpr dispatch_tokenize dispatch_tokenize{};
inline constexpr tokenize_error_backend tokenize_error_backend{};
inline constexpr set_error_invalid_argument set_error_invalid_argument{};
inline constexpr set_error_model_invalid set_error_model_invalid{};
inline constexpr set_error_capacity set_error_capacity{};
inline constexpr set_error_backend set_error_backend{};
inline constexpr set_error_untracked set_error_untracked{};
inline constexpr prepare_success prepare_success{};
inline constexpr write_bind_error_out write_bind_error_out{};
inline constexpr emit_bind_done emit_bind_done{};
inline constexpr emit_bind_error emit_bind_error{};
inline constexpr write_prepare_token_count write_prepare_token_count{};
inline constexpr write_prepare_error_out write_prepare_error_out{};
inline constexpr emit_prepare_done emit_prepare_done{};
inline constexpr emit_prepare_error emit_prepare_error{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::text::conditioner::action
