#pragma once

/*
design doc: docs/designs/graph.design.md
 ---
 title: graph architecture design
 status: draft
 ---
 
 # graph architecture design
 
 this document defines graph. the graph is an actor (sm) that acts as the compute
 manager for the generator. it owns the DAG topology — the nodes in topological
 order and the edges (source mappings).
 
 ## role
 - orchestrate the compute pipeline: delegate DAG construction to assembler, and
   execution to processor.
 - own the DAG topology and the Data-Oriented Design (DOD) tensor arrays: nodes
   in execution order, edges (source indices per node), and flat tensor metadata
   (`refs[]`, `states[]`, `pointers[]`).
 - provide traversal order for graph/processor.
 - manage tensor lifecycles procedurally: set `refs` during bind, allowing the
   processor to decrement them during execution without SML event overhead.
 
 ## composition
 - owned by generator.
 - owns graph/assembler (constructs or reuses the DAG).
 - owns graph/processor (executes the DAG via kernel).
 - owns the flat DOD tensor arrays (populated by assembler during reserve, managed
   here).
 
 ## state model (draft)
 
 ```text
 uninitialized ──► reserved ──► idle ──► executing ──► done
                                  ▲                      │
                                  │            ┌─────────┘
                                  │            ▼
                                  │   awaiting_barrier
                                  │            │
                                  └────────────┘
 ```
 
 - `uninitialized` — no graph built.
 - `reserved` — assembler has constructed worst-case DAG and allocated tensor
   buffers. ready for real assembles.
 - `idle` — graph ready, waiting for a compute request.
 - `executing` — processor is procedurally walking nodes and dispatching ops to kernel.
 - `awaiting_barrier` — the graph has dispatched work to a GPU processor and is waiting for the
   orchestrator to signal `event::barrier_complete`. the graph enters this state when the processor
   sets `requires_barrier = true` (GPU backends) and exits when it receives `barrier_complete`.
   CPU backends set `requires_barrier = false`, so the graph transitions directly from `executing`
   to `done` without entering this state.
 - `done` — execution complete, results in output tensors. transitions back
   to `idle`.
 
 ## events (draft)
 - `event::reserve` — from generator at init. forwards to assembler. inputs:
   model metadata (worst-case dims). transitions: `uninitialized → reserved`.
 - `event::compute` — from generator per step. inputs: `batch::plan`.
   forwards to assembler for assemble/reuse, then to processor for execution.
   transitions: `idle → executing`.
 - `event::barrier_complete` — sent by the orchestrator (via the generator) when GPU work finishes.
   inputs: `step_id`. transitions: `awaiting_barrier → idle`. this event is a no-op when no barrier
   is required and is rejected if `step_id` does not match the current in-flight step.
 - `events::compute_done` — execution complete. outputs: output tensor
   pointers (logits). transitions: `executing → done → idle`.
 - `events::compute_error` — error. outputs: error_out.
 
 ## DAG topology (DOD layout)
 - `nodes[]` — tensor indices in topological (execution) order.
 - `edges[]` — per node, source tensor indices (up to 4 sources: src0, src1,
   src2, src3). index into the DOD arrays.
 - `tensor_refs[]` — flat array of active consumer counts.
 - `tensor_states[]` — flat array of tensor lifecycle states (`allocated`, `empty`, `filled`).
 - `tensor_pointers[]` — flat array of hardware buffer pointers.
 
 ## ref management
 - on bind: graph walks `nodes[]`, counts consumers for each tensor
   (how many nodes list it as a source), and initializes `tensor_refs[id]` with the
   consumer count.
 - on op completion: the `graph/processor` directly decrements `tensor_refs[src_id]`.
   when a tensor's refs hit zero, its state returns to `empty` — the buffer region
   is available for reuse by the next tenant.
 
 ## requires_barrier flag
 
 the processor sets a `requires_barrier` flag based on the backend type:
 
 - CPU backends set `requires_barrier = false`. results are host-ready immediately after the
   execution loop completes, and the graph transitions straight to `done`.
 - GPU backends set `requires_barrier = true`. the GPU is still executing asynchronously after the
   encode loop returns, and the graph transitions to `awaiting_barrier` until the orchestrator
   signals completion.
 
 ## relationship to assembler and processor
 - assembler constructs/modifies the DAG topology and DOD arrays. the graph owns
   the topology after assemble.
 - processor reads the topology from the graph, walks nodes in a fast `for` loop,
   and uses the Opcode Router to dispatch each node's op to the kernel. after each
   op, it procedurally decrements `refs` on the sources.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_CAPACITY` — the requested allocation exceeds available tensor buffer or DAG capacity.
 - `EMEL_ERR_UNSUPPORTED_OP` — the kernel could not handle an opcode after fallback was exhausted.
 - `EMEL_ERR_INVALID_ARGUMENT` — the compute plan contained invalid bounds or references.
 - `EMEL_ERR_BUSY` — a compute request arrived while the graph is in `awaiting_barrier`.
 - `EMEL_ERR_INTERNAL` — an internal invariant was violated.
*/


// benchmark: scaffold
// docs: disabled

#include "emel/sm.hpp"
#include "emel/graph/events.hpp"

namespace emel::graph {

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

}  // namespace emel::graph
