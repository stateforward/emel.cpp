#pragma once

#include <cstdint>

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/errors.hpp"
#include "emel/text/encoders/events.hpp"

namespace emel::text::encoders::spm::action {

struct context : emel::text::encoders::action::context {
};

}  // namespace emel::text::encoders::spm::action

namespace emel::text::encoders::spm::runtime {

struct encode_runtime {
  const emel::text::encoders::event::encode_runtime & event_;
  mutable int32_t emit_result_error =
    emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  mutable int32_t emit_result_token_count = 0;
};

}  // namespace emel::text::encoders::spm::runtime
