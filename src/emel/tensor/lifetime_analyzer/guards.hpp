#pragma once

#include "emel/tensor/lifetime_analyzer/actions.hpp"

namespace emel::tensor::lifetime_analyzer::guard {

struct phase_ok {
  bool operator()(const action::context & c) const noexcept {
    return c.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & c) const noexcept {
    return c.phase_error != EMEL_OK;
  }
};

}  // namespace emel::tensor::lifetime_analyzer::guard
