#pragma once

#include "emel/error/error.hpp"
#include "emel/speech/recognizer/errors.hpp"

namespace emel::speech::recognizer::detail {

inline emel::error::type to_error(const error err) noexcept {
  return emel::error::cast(err);
}

}  // namespace emel::speech::recognizer::detail
