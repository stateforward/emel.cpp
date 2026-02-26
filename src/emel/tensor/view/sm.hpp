#pragma once

/*
design doc: docs/designs/tensor/view.design.md
 ---
 title: tensor/view architecture design
 status: draft
 ---
 
 # tensor/view architecture design
 
 this document defines tensor/view. a view shares its source tensor's buffer at an offset — acting as
 a hardlink to the source's data blocks.
 
 ## role
 - provide an alternative interpretation (shape, stride, offset) of a source tensor's buffer without
   owning separate storage.
 - hold a ref (hardlink) on the source tensor to prevent the source from going empty while the view
   is active.
 - track its own consumer refs, just like a regular tensor.
 
 ## architecture shift: data-oriented design (`sml::sm_pool`)
 like regular tensors, views are managed using `boost::sml::sm_pool` to eliminate SML dispatch
 overhead in the hot path while retaining strict actor model safety boundaries.
 their lifecycle is managed via the blazing-fast `process_event_batch` API by the `graph/processor`.
 
 ## inode analogy
 - the view is a hardlink to the source inode's data blocks.
 - the source's link count includes the view — the source can't be freed while the hardlink exists.
 - the view has its own link count from its consumers.
 
 ## state model (logical)
 
 ```text
 unlinked ──► linked ──► empty ──► filled
                           ▲          │
                           │   refs--  │
                           │          │
                           └──────────┘
                            refs == 0
 ```
 
 - `unlinked` — no source. view knows its shape/stride/offset but is not connected to a source.
 - `linked` — source ref acquired. view holds a pointer derived from source pointer + offset.
 - `empty` — linked to source, no live data from this view's perspective. ready to be used by consumers.
 - `filled` — source data is live, consumers reading through this view. refs tracks active consumers.
   when refs == 0, transitions back to `empty`.
 
 ## link lifecycle
 - **binding:** the graph records the link, and the view increments the source's refs.
 - **unlinking:** when the view's own refs hit 0 (all its consumers are done), the processor decrements
   the source tensor's ref count, effectively releasing the view's hardlink.
 - the source tensor cannot go empty while any view holds a ref on it.
 
 ## differences from tensor
 - no allocation phase — view does not reserve a buffer. its pointer is derived from the source's
   pointer + offset.
 - holds a permanent ref on its source (released when execution cycle ends or graph is torn down).
 - ops are shape manipulations (view, reshape, permute, transpose) — no computation, just
   reinterpretation of existing data.
 
 ## composition
 - views are stored alongside regular tensors in the graph topology.
 - the graph topology tracks view → source edges separately from op source edges.
 - the `graph/processor` sets the view to `filled` when the source tensor is filled (the view's "op"
   is essentially a no-op — the data is already there).
 
 ## constraints
 - a view always links to a tensor, never to another view. multiple views of the same tensor at
   different offsets are independent hardlinks to the same inode.
*/


// benchmark: scaffold
// docs: disabled

#include "emel/sm.hpp"
#include "emel/tensor/view/events.hpp"

namespace emel::tensor::view {

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

}  // namespace emel::tensor::view
