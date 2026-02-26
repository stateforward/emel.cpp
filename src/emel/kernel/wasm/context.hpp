#pragma once

#include <cstdint>

namespace emel::kernel::wasm::action {

struct context {
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::wasm::action
