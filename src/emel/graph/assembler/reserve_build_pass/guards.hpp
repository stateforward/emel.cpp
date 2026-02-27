#pragma once

#include <cstdint>
#include <limits>

#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/reserve_build_pass/context.hpp"
#include "emel/graph/assembler/reserve_validate_pass/events.hpp"

namespace emel::graph::assembler::reserve_build_pass::guard {

inline bool product_overflows_u64(const uint64_t lhs, const uint64_t rhs) noexcept {
  return lhs != 0u && rhs > (std::numeric_limits<uint64_t>::max() / lhs);
}

struct phase_done {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.validate_outcome == reserve_validate_pass::events::phase_outcome::done &&
           ev.request.max_node_count != 0u &&
           ev.request.max_tensor_count != 0u &&
           ev.request.bytes_per_tensor != 0u &&
           !product_overflows_u64(static_cast<uint64_t>(ev.request.max_tensor_count),
                                  ev.request.bytes_per_tensor) &&
           static_cast<uint64_t>(ev.request.max_tensor_count) * ev.request.bytes_per_tensor <=
               ev.request.workspace_capacity_bytes;
  }
};

struct phase_prereq_failed {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.validate_outcome != reserve_validate_pass::events::phase_outcome::done;
  }
};

struct phase_capacity_exceeded {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.validate_outcome == reserve_validate_pass::events::phase_outcome::done &&
           ev.request.max_tensor_count != 0u &&
           ev.request.bytes_per_tensor != 0u &&
           ((!product_overflows_u64(static_cast<uint64_t>(ev.request.max_tensor_count),
                                    ev.request.bytes_per_tensor) &&
             static_cast<uint64_t>(ev.request.max_tensor_count) * ev.request.bytes_per_tensor >
                 ev.request.workspace_capacity_bytes) ||
            product_overflows_u64(static_cast<uint64_t>(ev.request.max_tensor_count),
                                  ev.request.bytes_per_tensor));
  }
};

struct phase_invalid_request {
  bool operator()(const assembler::event::reserve_graph & ev, const action::context &) const noexcept {
    return ev.ctx.err == emel::error::cast(assembler::error::none) &&
           ev.ctx.validate_outcome == reserve_validate_pass::events::phase_outcome::done &&
           (ev.request.max_node_count == 0u ||
            ev.request.max_tensor_count == 0u ||
            ev.request.bytes_per_tensor == 0u);
  }
};

}  // namespace emel::graph::assembler::reserve_build_pass::guard
