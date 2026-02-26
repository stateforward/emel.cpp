#pragma once

/*
design doc: docs/designs/kernel/x86_64.design.md
 ---
 title: kernel/x86_64 architecture design
 status: draft
 ---
 
 # kernel/x86_64 architecture design
 
 this document defines kernel/x86_64. it executes typed kernel op events on x86-64 cpus.
 
 ## role
 - execute `op::*` events on x86-64 hosts.
 - select best ISA tier per op at bind time via CPUID (AVX-512 > AVX2 > scalar).
 
 ## events (draft)
 - `event::bind` inputs: cpu execution policy and hardware context.
 - `op::*` inputs: destination/source tensor handles plus shape/stride/op metadata.
 - outputs: writes op results in-place; unsupported ops route through `sml::unexpected_event`.
 
 ## state model (draft)
 - `uninitialized` -> `binding` -> `idle`.
 - `idle` handles incoming `op::*` events.
 - unexpected non-op events route to `unexpected`.
 
 ## responsibilities
 - populate per-opcode dispatch table from CPUID feature detection.
 - execute each received `op::*` event using AVX-512, AVX2, or scalar fallback.
 - reuse scratch buffers across op events.
 - avoid graph traversal here; node walking stays in `graph/processor`.
*/


// benchmark: scaffold
// docs: disabled

#include "emel/kernel/events.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::x86_64 {

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

}  // namespace emel::kernel::x86_64
