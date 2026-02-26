#pragma once

/*
design doc: docs/designs/graph/allocator.design.md
 ---
 title: graph/allocator architecture design
 status: draft
 ---
 
 # graph/allocator architecture design
 
 this document defines the graph/allocator. it replaces dynamic chunk allocation with
 static interval coloring, assigning memory offsets to tensor actors based on their DAG lifecycle.
 
 ## role
 - analyze a DAG topology to determine the exact lifecycle (liveness interval) of every tensor.
 - assign non-overlapping static memory offsets to all compute tensors.
 - calculate the total required continuous buffer size for the entire graph.
 - provide a purely declarative, mathematical planning phase to the `graph/assembler`.
 
 ## architecture shift: from dynamic to static
 unlike traditional procedural allocators (like `ggml_dyn_tallocr` or the deprecated
 `buffer/chunk_allocator`) which simulate runtime `malloc`/`free` operations and merge free blocks,
 the graph/allocator treats memory assignment as an **interval graph coloring problem**.
 
 because the DAG execution order is deterministic, the allocator knows exactly when a tensor is
 created (first write) and when it is no longer needed (last read). memory blocks are never
 "freed" at runtime; offsets are simply reused by later tensors whose lifecycles do not overlap.
 
 this static approach completely eliminates the need for complex allocator state machines,
 reducing memory planning to a fast `O(N log N)` algorithmic pass.
 
 ## responsibilities
 
 1. **liveness analysis:**
    - walk the DAG nodes in execution order.
    - for each tensor, determine its `start_node` (the node that produces it).
    - find its `end_node` (the latest node in the topological order that consumes it as a source).
    - the tensor's liveness interval is `[start_node, end_node]`.
 
 2. **offset assignment (interval allocation):**
    - sort tensors by their `start_node`.
    - maintain a list of active memory intervals (offset, size, `end_node`).
    - for each tensor, find the lowest memory offset that satisfies its alignment requirements
      and does not overlap with any currently active interval.
    - assign that offset to the tensor.
    - evict any intervals from the active list whose `end_node` is strictly less than the current
      tensor's `start_node`.
 
 3. **total sizing:**
    - the maximum address reached during assignment (highest offset + size) becomes the
      total required buffer size for the graph.
 
 ## relationship to assembler and tensors
 - **`graph/assembler`**: during the `reserve` phase (for worst-case sizing) or when a topology
   changes, the assembler constructs the DAG and hands it to the `graph/allocator`.
 - the `graph/allocator` returns the computed offsets and the total required size.
 - the `graph/assembler` uses these offsets to assign the final memory pointers to the `tensor` actors.
 - **`tensor` actors**: during execution, the tensor actors' SML guards (`refs == 0`) enforce the
   liveness intervals calculated by the allocator. if the allocator algorithm has a bug and overlaps
   two live tensors, the tensor's state machine will crash safely (receiving a `fill` while `filled`),
   preventing silent memory corruption.
 
 ## execution frequency (watermark strategy)
 the allocator typically runs **once per session** during the `reserve` phase to calculate the
 worst-case memory watermark. during the per-batch `compute` phase, if the DAG topology matches the
 worst-case structure (which is true for standard transformer architectures), the assembler skips the
 allocator entirely and reuses the static offsets. the allocator only runs again if the actual
 execution graph changes its shape (e.g., dynamic routing in MoE models).
 
 ## determinism
 
 tensor sorting uses a strict total order with explicit tie-breakers. the primary sort key is tensor
 size (descending). ties are broken by tensor name (lexicographic ascending). given identical graph
 structure and identical tensor metadata, allocation layouts are identical across runs. this
 deterministic ordering ensures reproducible memory maps for debugging.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_CAPACITY` — reserve-time capacity is insufficient, for example when scratch buffers are too small or the tensor count exceeds the reserved maximum.
*/


// benchmark: scaffold
// docs: disabled

#include "emel/sm.hpp"
#include "emel/graph/allocator/events.hpp"

namespace emel::graph::allocator {

struct idle {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::scaffold> = sml::state<idle>,
      sml::state<idle> + sml::unexpected_event<sml::_> = sml::state<idle>
    );
  }
};

using sm = emel::sm<model>;

}  // namespace emel::graph::allocator
