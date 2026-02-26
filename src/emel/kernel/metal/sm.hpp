#pragma once

/*
design doc: docs/designs/kernel/metal.design.md
 ---
 title: kernel/metal architecture design
 status: draft
 ---
 
 # kernel/metal architecture design
 
 this document defines kernel/metal. it executes typed kernel op events on
 Apple Metal GPUs.
 
 ## role
 - execute `op::*` events via Metal compute shaders.
 - available on macOS and iOS with Metal support.
 
 ## events (draft)
 - `event::bind` inputs: gpu execution policy and device context.
 - `op::*` inputs: destination/source tensor handles plus shape/stride/op metadata.
 - outputs: writes op results in-place; unsupported ops route through `sml::unexpected_event`.
 
 ## state model (draft)
 - `uninitialized` -> `binding` -> `idle`.
 - `idle` handles incoming `op::*` events.
 - unexpected non-op events route to `unexpected`.
 
 ## responsibilities
 - map opcodes to Metal compute pipeline states.
 - encode commands into a command buffer for each op event.
 - manage Metal buffer bindings per op event.
 - the Metal kernel actor encodes operations into a command buffer. it does not commit or wait after
   each operation. command buffer commit and `waitUntilCompleted` happen at barrier time, outside
   the kernel dispatch path.
 
 no blocking calls are permitted inside dispatch actions.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_UNSUPPORTED_OP` — the opcode is not supported by this backend.
 - `EMEL_ERR_CAPACITY` — command encoding exceeded the preallocated resource capacity.
*/


// benchmark: scaffold
// docs: disabled

#include "emel/kernel/events.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::metal {

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

}  // namespace emel::kernel::metal
