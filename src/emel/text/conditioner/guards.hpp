#pragma once

#include "emel/text/conditioner/context.hpp"
#include "emel/text/conditioner/events.hpp"

namespace emel::text::conditioner::guard {

struct valid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    if (ev.vocab == nullptr || ev.tokenizer_sm == nullptr ||
        ev.dispatch_tokenizer_bind == nullptr ||
        ev.dispatch_tokenizer_tokenize == nullptr ||
        ev.format_prompt == nullptr) {
      return false;
    }
    switch (ev.preprocessor_variant) {
      case emel::text::tokenizer::preprocessor::preprocessor_kind::spm:
      case emel::text::tokenizer::preprocessor::preprocessor_kind::bpe:
      case emel::text::tokenizer::preprocessor::preprocessor_kind::wpm:
      case emel::text::tokenizer::preprocessor::preprocessor_kind::ugm:
      case emel::text::tokenizer::preprocessor::preprocessor_kind::rwkv:
      case emel::text::tokenizer::preprocessor::preprocessor_kind::plamo2:
      case emel::text::tokenizer::preprocessor::preprocessor_kind::fallback:
        break;
      default:
        return false;
    }
    switch (ev.encoder_variant) {
      case emel::text::encoders::encoder_kind::spm:
      case emel::text::encoders::encoder_kind::bpe:
      case emel::text::encoders::encoder_kind::wpm:
      case emel::text::encoders::encoder_kind::ugm:
      case emel::text::encoders::encoder_kind::rwkv:
      case emel::text::encoders::encoder_kind::plamo2:
      case emel::text::encoders::encoder_kind::fallback:
        break;
      default:
        return false;
    }
    return true;
  }
};

struct invalid_bind {
  bool operator()(const event::bind & ev) const noexcept {
    return !valid_bind{}(ev);
  }
};

struct valid_prepare {
  bool operator()(const event::prepare & ev,
                  const action::context & ctx) const noexcept {
    return ctx.is_bound && ctx.vocab != nullptr && ev.token_ids_out != nullptr &&
           ev.token_count_out != nullptr && ev.error_out != nullptr &&
           ev.token_capacity > 0;
  }
};

struct invalid_prepare {
  bool operator()(const event::prepare & ev,
                  const action::context & ctx) const noexcept {
    return !valid_prepare{}(ev, ctx);
  }
};

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

}  // namespace emel::text::conditioner::guard
