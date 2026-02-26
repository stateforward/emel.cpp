#pragma once

/*
design doc: docs/designs/kernel/cuda.design.md
 ---
 title: kernel/cuda architecture design
 status: draft
 ---
 
 # kernel/cuda architecture design
 
 this document defines kernel/cuda. it executes typed kernel op events on
 NVIDIA GPUs via CUDA.
 
 ## role
 - execute `op::*` events via CUDA kernels.
 - available on platforms with NVIDIA CUDA support.
 
 ## events (draft)
 - `event::bind` inputs: gpu execution policy and device context.
 - `op::*` inputs: destination/source tensor handles plus shape/stride/op metadata.
 - outputs: writes op results in-place; unsupported ops route through `sml::unexpected_event`.
 
 ## state model (draft)
 - `uninitialized` -> `binding` -> `idle`.
 - `idle` handles incoming `op::*` events.
 - unexpected non-op events route to `unexpected`.
 
 ## responsibilities
 - map opcodes to CUDA kernel launches.
 - manage CUDA stream and device buffer bindings per op event.
 - encode operations into a CUDA stream without synchronizing after each operation. the actor
   enqueues kernel launches into the stream and returns immediately. stream synchronization
   happens at barrier time, outside the kernel dispatch path.
 
 no blocking calls (`cudaStreamSynchronize`, `cudaDeviceSynchronize`, or event/fence waits) are
 permitted inside dispatch actions.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_UNSUPPORTED_OP` — the opcode is not supported by this backend.
 - `EMEL_ERR_CAPACITY` — the command recording exceeded the preallocated command buffer capacity.
*/


// benchmark: scaffold
// docs: disabled

#include "emel/kernel/events.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::cuda {

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

}  // namespace emel::kernel::cuda
