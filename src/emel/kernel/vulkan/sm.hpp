#pragma once

/*
design doc: docs/designs/kernel/vulkan.design.md
 ---
 title: kernel/vulkan architecture design
 status: draft
 ---
 
 # kernel/vulkan architecture design
 
 this document defines kernel/vulkan. it executes typed kernel op events on
 Vulkan-capable GPUs.
 
 ## role
 - execute `op::*` events via Vulkan compute shaders.
 - available on platforms with Vulkan support (Linux, Windows, Android).
 
 ## events (draft)
 - `event::bind` inputs: gpu execution policy and device context.
 - `op::*` inputs: destination/source tensor handles plus shape/stride/op metadata.
 - outputs: writes op results in-place; unsupported ops route through `sml::unexpected_event`.
 
 ## state model (draft)
 - `uninitialized` -> `binding` -> `idle`.
 - `idle` handles incoming `op::*` events.
 - unexpected non-op events route to `unexpected`.
 
 ## responsibilities
 - map opcodes to Vulkan compute pipeline objects.
 - record commands into a command buffer for each op event.
 - manage Vulkan buffer bindings and descriptor sets per op event.
 - the Vulkan kernel actor records operations into a command buffer. it does not submit or
   fence-wait after each operation. queue submit and fence wait happen at barrier time, outside
   the kernel dispatch path.
 
 no blocking calls (`vkQueueSubmit` + `vkWaitForFences`, `vkQueueWaitIdle`) are permitted inside
 dispatch actions.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_UNSUPPORTED_OP` — the opcode is not supported by this backend.
 - `EMEL_ERR_CAPACITY` — command recording exceeded the preallocated command buffer capacity.
*/


// benchmark: scaffold
// docs: disabled

#include "emel/kernel/events.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::vulkan {

struct idle {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<idle> + sml::event<::emel::kernel::event::scaffold> = sml::state<idle>,
      sml::state<idle> + sml::unexpected_event<sml::_> = sml::state<idle>
    );
  }
};

using sm = emel::sm<model>;

}  // namespace emel::kernel::vulkan
