#pragma once

#include "emel/kernel/wasm/context.hpp"

namespace emel::kernel::wasm::action {

struct noop {
  void operator()(context &) const noexcept {}
};

}  // namespace emel::kernel::wasm::action
