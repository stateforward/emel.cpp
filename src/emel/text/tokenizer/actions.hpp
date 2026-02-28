#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/text/encoders/events.hpp"
#include "emel/text/tokenizer/context.hpp"
#include "emel/text/tokenizer/detail.hpp"
#include "emel/text/tokenizer/errors.hpp"
#include "emel/text/tokenizer/events.hpp"
#include "emel/text/tokenizer/preprocessor/events.hpp"

namespace emel::text::tokenizer::action {

inline context::context() : encoder_any() {
  preprocessor_any.set_kind(preprocessor_kind::fallback);
  encoder_any.set_kind(encoder_kind::fallback);
  preprocess_kind = preprocessor_kind::fallback;
  model_kind = encoder_kind::fallback;
  is_bound = false;
}

namespace detail {

template <class runtime_event_type>
inline void append_token(const runtime_event_type &runtime_ev,
                         const int32_t token) noexcept {
  auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
  ev.request.token_ids_out[ev.ctx.token_count] = token;
  ev.ctx.token_count += 1;
}

template <class runtime_event_type>
inline void set_error(const runtime_event_type &runtime_ev,
                      const int32_t err) noexcept {
  auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
  ev.ctx.err = err;
  ev.ctx.result = false;
}

template <class runtime_event_type>
inline void clear_bind_runtime(const runtime_event_type &runtime_ev) noexcept {
  auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
  ev.ctx.err = error_code(error::none);
  ev.ctx.result = false;
}

template <class runtime_event_type>
inline void
clear_tokenize_runtime(const runtime_event_type &runtime_ev) noexcept {
  auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
  ev.ctx.fragment_count = 0;
  ev.ctx.fragment_index = 0;
  ev.ctx.preprocessed = false;
  ev.ctx.preprocess_accepted = false;
  ev.ctx.preprocess_err_code = error_code(error::none);
  ev.ctx.encode_accepted = false;
  ev.ctx.encode_err_code = error_code(error::none);
  ev.ctx.encode_token_count = 0;
  ev.ctx.token_count = 0;
  ev.ctx.err = error_code(error::none);
  ev.ctx.result = false;
}

} // namespace detail

struct begin_bind {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev =
        emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    detail::clear_bind_runtime(ev);
    ctx.vocab = ev.request.vocab;
    ctx.preprocess_kind = ev.request.preprocessor_variant;
    ctx.model_kind = ev.request.encoder_variant;
    ctx.is_bound = false;
  }
};

struct reject_bind {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    ctx.is_bound = false;
    detail::set_error(runtime_ev, error_code(error::invalid_request));
  }
};

struct bind_preprocessor {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    ctx.preprocessor_any.set_kind(ctx.preprocess_kind);
    ev.ctx.err = error_code(error::none);
  }
};

struct bind_encoder {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    ctx.is_bound = false;
    ctx.encoder_any.set_kind(ctx.model_kind);
    ev.ctx.err = error_code(error::none);
  }
};

struct mark_bind_success {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    ctx.is_bound = true;
    ev.ctx.err = error_code(error::none);
    ev.ctx.result = true;
  }
};

struct begin_tokenize {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    detail::clear_tokenize_runtime(runtime_ev);
  }
};

struct reject_invalid {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.token_count = 0;
    detail::set_error(ev, error_code(error::invalid_request));
  }
};

struct dispatch_preprocess {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    size_t fragment_count = 0;
    bool preprocessed = false;
    int32_t err = error_code(error::none);

    emel::text::tokenizer::preprocessor::event::preprocess pre_ev = {};
    pre_ev.vocab = ctx.vocab;
    pre_ev.text = ev.request.text;
    pre_ev.parse_special = ev.request.parse_special;
    pre_ev.fragments_out = ev.ctx.fragments.data();
    pre_ev.fragment_capacity = ev.ctx.fragments.size();
    pre_ev.fragment_count_out = &fragment_count;
    pre_ev.preprocessed_out = &preprocessed;
    pre_ev.error_out = &err;

    ev.ctx.preprocess_accepted = ctx.preprocessor_any.process_event(pre_ev);
    ev.ctx.preprocess_err_code = err;
    ev.ctx.fragment_count = fragment_count;
    ev.ctx.fragment_index = 0;
    ev.ctx.preprocessed = preprocessed;
  }
};

struct set_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    detail::set_error(runtime_ev, error_code(error::backend_error));
  }
};

struct set_error_from_preprocess {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    detail::set_error(ev, ev.ctx.preprocess_err_code);
  }
};

struct append_bos {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    detail::append_token(ev, ctx.vocab->bos_id);
    ev.ctx.err = error_code(error::none);
  }
};

struct append_sep {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    detail::append_token(ev, ctx.vocab->sep_id);
    ev.ctx.err = error_code(error::none);
  }
};

struct append_eos {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    detail::append_token(ev, ctx.vocab->eos_id);
    ev.ctx.err = error_code(error::none);
  }
};

struct append_fragment_token {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    const fragment &frag = ev.ctx.fragments[ev.ctx.fragment_index];
    detail::append_token(ev, frag.token);
    ev.ctx.fragment_index += 1;
    ev.ctx.err = error_code(error::none);
  }
};

struct dispatch_encode_raw_fragment {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    const fragment &frag = ev.ctx.fragments[ev.ctx.fragment_index];
    const int32_t capacity = ev.request.token_capacity - ev.ctx.token_count;

    int32_t fragment_count = 0;
    int32_t err = error_code(error::none);
    emel::text::encoders::event::encode encode_ev = {};
    encode_ev.vocab = ctx.vocab;
    encode_ev.text = frag.text;
    encode_ev.preprocessed = ev.ctx.preprocessed;
    encode_ev.token_ids = ev.request.token_ids_out + ev.ctx.token_count;
    encode_ev.token_capacity = capacity;
    encode_ev.token_count_out = &fragment_count;
    encode_ev.error_out = &err;

    ev.ctx.encode_accepted = ctx.encoder_any.process_event(encode_ev);
    ev.ctx.encode_err_code = err;
    ev.ctx.encode_token_count = fragment_count;
  }
};

struct set_error_from_encode {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    detail::set_error(ev, ev.ctx.encode_err_code);
  }
};

struct commit_encoded_fragment {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.token_count += ev.ctx.encode_token_count;
    ev.ctx.fragment_index += 1;
    ev.ctx.err = error_code(error::none);
  }
};

struct finalize {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = error_code(error::none);
    ev.ctx.result = true;
  }
};

struct set_invalid_request_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    detail::set_error(runtime_ev, error_code(error::invalid_request));
  }
};

struct set_invalid_id_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    detail::set_error(runtime_ev, error_code(error::model_invalid));
  }
};

struct on_unexpected {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    auto &ev = emel::text::tokenizer::detail::unwrap_runtime_event(runtime_ev);
    if constexpr (requires { ev.ctx.token_count; }) {
      ev.ctx.token_count = 0;
    }
    ev.ctx.err = error_code(error::invalid_request);
    ev.ctx.result = false;
  }
};

inline constexpr begin_bind begin_bind{};
inline constexpr reject_bind reject_bind{};
inline constexpr bind_preprocessor bind_preprocessor{};
inline constexpr bind_encoder bind_encoder{};
inline constexpr mark_bind_success mark_bind_success{};
inline constexpr begin_tokenize begin_tokenize{};
inline constexpr reject_invalid reject_invalid{};
inline constexpr dispatch_preprocess dispatch_preprocess{};
inline constexpr set_backend_error set_backend_error{};
inline constexpr set_error_from_preprocess set_error_from_preprocess{};
inline constexpr append_bos append_bos{};
inline constexpr append_sep append_sep{};
inline constexpr append_eos append_eos{};
inline constexpr append_fragment_token append_fragment_token{};
inline constexpr dispatch_encode_raw_fragment dispatch_encode_raw_fragment{};
inline constexpr set_error_from_encode set_error_from_encode{};
inline constexpr commit_encoded_fragment commit_encoded_fragment{};
inline constexpr finalize finalize{};
inline constexpr set_invalid_request_error set_invalid_request_error{};
inline constexpr set_invalid_id_error set_invalid_id_error{};
inline constexpr on_unexpected on_unexpected{};

} // namespace emel::text::tokenizer::action
