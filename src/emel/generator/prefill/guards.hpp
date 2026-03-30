#pragma once

#include "emel/generator/context.hpp"
#include "emel/generator/detail.hpp"
#include "emel/generator/prefill/context.hpp"
#include "emel/generator/prefill/detail.hpp"
#include "emel/graph/errors.hpp"
#include "emel/memory/hybrid/errors.hpp"

namespace emel::generator::prefill::guard {

namespace detail {

inline bool has_phase_success(const event::run & ev) noexcept {
  return ev.ctx.phase_accepted && ev.ctx.phase_code == 0;
}

inline bool phase_rejected_without_code(const event::run & ev) noexcept {
  return !ev.ctx.phase_accepted && ev.ctx.phase_code == 0;
}

constexpr int32_t memory_code(const emel::memory::hybrid::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t graph_code(const emel::graph::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

inline bool memory_invalid_code(const int32_t code) noexcept {
  return code == memory_code(emel::memory::hybrid::error::invalid_request);
}

inline bool memory_backend_code(const int32_t code) noexcept {
  return code == memory_code(emel::memory::hybrid::error::backend_error) ||
         code == memory_code(emel::memory::hybrid::error::internal_error) ||
         code == memory_code(emel::memory::hybrid::error::out_of_memory) ||
         code == memory_code(emel::memory::hybrid::error::untracked);
}

inline bool graph_invalid_code(const int32_t code) noexcept {
  return code == graph_code(emel::graph::error::invalid_request);
}

inline bool graph_backend_code(const int32_t code) noexcept {
  return code == graph_code(emel::graph::error::assembler_failed) ||
         code == graph_code(emel::graph::error::processor_failed) ||
         code == graph_code(emel::graph::error::busy) ||
         code == graph_code(emel::graph::error::internal_error) ||
         code == graph_code(emel::graph::error::untracked);
}

inline bool uses_preselected_argmax_direct(const action::context & ctx) noexcept {
  return ctx.generator.state.selection_mode == emel::generator::selection_mode::preselected_argmax &&
         emel::generator::detail::preselected_argmax_direct_supported(
             ctx.generator.compute.backend);
}

inline bool uses_prefill_chunk4_q8_gemm(const event::run & ev,
                                        const action::context & ctx) noexcept {
  return ev.ctx.prompt_token_count >= emel::generator::detail::k_prefill_q8_chunk_rows &&
         emel::generator::detail::prefill_chunk4_q8_gemm_supported(
             ctx.generator.compute.backend);
}

}  // namespace detail

struct slots_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct slots_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::memory_invalid_code(ev.ctx.phase_code);
  }
};

struct slots_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct snapshot_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct snapshot_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::memory_invalid_code(ev.ctx.phase_code);
  }
};

struct snapshot_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct flash_runtime_supported {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return ev.ctx.prompt_token_count > 0 &&
           emel::generator::detail::flash_attention_supported(
               ctx.generator.compute.backend, ev.ctx.prompt_token_count - 1);
  }
};

struct nonflash_runtime_required {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !flash_runtime_supported{}(ev, ctx);
  }
};

struct uses_materialized_logits_with_chunk4 {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !detail::uses_preselected_argmax_direct(ctx) && detail::uses_prefill_chunk4_q8_gemm(ev, ctx);
  }
};

struct uses_materialized_logits_with_scalar {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !detail::uses_preselected_argmax_direct(ctx) && !detail::uses_prefill_chunk4_q8_gemm(ev, ctx);
  }
};

struct uses_preselected_argmax_with_chunk4 {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_preselected_argmax_direct(ctx) && detail::uses_prefill_chunk4_q8_gemm(ev, ctx);
  }
};

struct uses_preselected_argmax_with_scalar {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_preselected_argmax_direct(ctx) && !detail::uses_prefill_chunk4_q8_gemm(ev, ctx);
  }
};

struct compute_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct compute_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::graph_invalid_code(ev.ctx.phase_code);
  }
};

struct compute_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::graph_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::graph_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

}  // namespace emel::generator::prefill::guard
