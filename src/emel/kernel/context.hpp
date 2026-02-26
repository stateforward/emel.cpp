#pragma once

#include <cstdint>

#include "emel/kernel/aarch64/sm.hpp"
#include "emel/kernel/cuda/sm.hpp"
#include "emel/kernel/metal/sm.hpp"
#include "emel/kernel/vulkan/sm.hpp"
#include "emel/kernel/wasm/sm.hpp"
#include "emel/kernel/x86_64/sm.hpp"

namespace emel::kernel::action {

struct context {
  x86_64::sm x86_64_actor = {};
  aarch64::sm aarch64_actor = {};
  wasm::sm wasm_actor = {};
  cuda::sm cuda_actor = {};
  metal::sm metal_actor = {};
  vulkan::sm vulkan_actor = {};
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::action
