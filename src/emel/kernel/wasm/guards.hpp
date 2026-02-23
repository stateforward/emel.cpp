#pragma once

#include "emel/kernel/wasm/context.hpp"

namespace emel::kernel::wasm::guard {

struct always {
  bool operator()(const action::context &) const noexcept {
    return true;
  }
};

}  // namespace emel::kernel::wasm::guard
