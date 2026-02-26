#pragma once

#include <cstdint>
#include <limits>

#include "emel/graph/assembler/assemble_build_pass/context.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/reuse_decision_pass/events.hpp"

namespace emel::graph::assembler::assemble_build_pass::guard {

inline bool product_overflows_u64(const uint64_t lhs, const uint64_t rhs) noexcept {
  return lhs != 0u && rhs > (std::numeric_limits<uint64_t>::max() / lhs);
}

struct phase_done {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.reuse_outcome == reuse_decision_pass::events::phase_outcome::rebuild &&
           ev.ctx.assembled_node_count != 0u &&
           ev.ctx.assembled_tensor_count != 0u &&
           ev.request.bytes_per_tensor != 0u &&
           !product_overflows_u64(static_cast<uint64_t>(ev.ctx.assembled_tensor_count),
                                  ev.request.bytes_per_tensor) &&
           static_cast<uint64_t>(ev.ctx.assembled_tensor_count) * ev.request.bytes_per_tensor <=
               ev.request.workspace_capacity_bytes;
  }
};

struct phase_prereq_failed {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.reuse_outcome != reuse_decision_pass::events::phase_outcome::rebuild;
  }
};

struct phase_capacity_exceeded {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.reuse_outcome == reuse_decision_pass::events::phase_outcome::rebuild &&
           ev.ctx.assembled_tensor_count != 0u &&
           ev.request.bytes_per_tensor != 0u &&
           ((!product_overflows_u64(static_cast<uint64_t>(ev.ctx.assembled_tensor_count),
                                    ev.request.bytes_per_tensor) &&
             static_cast<uint64_t>(ev.ctx.assembled_tensor_count) * ev.request.bytes_per_tensor >
                 ev.request.workspace_capacity_bytes) ||
            product_overflows_u64(static_cast<uint64_t>(ev.ctx.assembled_tensor_count),
                                  ev.request.bytes_per_tensor));
  }
};

struct phase_invalid_request {
  bool operator()(const assembler::event::assemble_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.reuse_outcome == reuse_decision_pass::events::phase_outcome::rebuild &&
           (ev.ctx.assembled_node_count == 0u ||
            ev.ctx.assembled_tensor_count == 0u ||
            ev.request.bytes_per_tensor == 0u);
  }
};

}  // namespace emel::graph::assembler::assemble_build_pass::guard
