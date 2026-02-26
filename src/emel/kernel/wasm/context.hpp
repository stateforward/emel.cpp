#pragma once

#include <cstdint>

namespace emel::kernel::wasm::action {

struct context {
  // TODO(emel): remove once scaffold observability no longer relies on this counter.
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::wasm::action
