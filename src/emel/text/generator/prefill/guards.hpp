#pragma once

#include "emel/text/generator/context.hpp"
#include "emel/text/generator/detail.hpp"
#include "emel/text/generator/guards.hpp"
#include "emel/text/generator/prefill/context.hpp"
#include "emel/text/generator/prefill/detail.hpp"
#include "emel/graph/errors.hpp"
#include "emel/memory/hybrid/errors.hpp"

namespace emel::text::generator::prefill::guard {

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
  return ctx.generator.state.selection_mode == emel::text::generator::selection_mode::preselected_argmax &&
         emel::text::generator::guard::detail::preselected_argmax_direct_supported(
             ctx.generator.compute.backend);
}

inline bool uses_parallel_matmul_lanes(const event::run & ev,
                                       const action::context & ctx) noexcept {
  return ctx.generator.compute.backend.lane_pool.has_value() &&
         ev.ctx.prompt_token_count >=
             emel::text::generator::detail::k_parallel_min_prefill_tokens;
}

inline bool uses_prefill_chunk4_q8_gemm(const event::run & ev,
                                        const action::context & ctx) noexcept {
  return ev.ctx.prompt_token_count >= emel::text::generator::detail::k_prefill_q8_chunk_rows &&
         emel::text::generator::guard::detail::prefill_chunk4_q8_gemm_supported(
             ctx.generator.compute.backend);
}

inline bool uses_prefill_chunk8_q8_k_gemm(const event::run & ev,
                                          const action::context & ctx) noexcept {
  return ev.ctx.prompt_token_count >= emel::text::generator::detail::k_prefill_q8_chunk8_rows &&
         emel::text::generator::guard::detail::prefill_chunk8_q8_k_supported(
             ctx.generator.compute.backend);
}

inline bool uses_prefill_chunk4_packed_q8_0_gemm(const event::run & ev,
                                                 const action::context & ctx) noexcept {
  return ev.ctx.prompt_token_count >= emel::text::generator::detail::k_prefill_q8_chunk_rows &&
         emel::text::generator::guard::detail::prefill_chunk4_packed_q8_0_supported(
             ctx.generator.compute.backend);
}

inline bool uses_prefill_chunk4_q8_k_gemm(const event::run & ev,
                                          const action::context & ctx) noexcept {
  return ev.ctx.prompt_token_count >= emel::text::generator::detail::k_prefill_q8_chunk_rows &&
         emel::text::generator::guard::detail::prefill_chunk4_q8_k_supported(
             ctx.generator.compute.backend);
}

inline bool uses_scalar_packed_q8_0(const action::context & ctx) noexcept {
  return emel::text::generator::guard::detail::scalar_matmul_packed_q8_0_supported(
      ctx.generator.compute.backend);
}

inline bool uses_scalar_q8_k(const action::context & ctx) noexcept {
  return !uses_scalar_packed_q8_0(ctx) &&
      emel::text::generator::guard::detail::scalar_matmul_q8_k_supported(
          ctx.generator.compute.backend);
}

inline bool uses_scalar_native_quantized(const action::context & ctx) noexcept {
  return !uses_scalar_packed_q8_0(ctx) &&
      !uses_scalar_q8_k(ctx) &&
      emel::text::generator::guard::detail::scalar_matmul_native_quantized_supported(
          ctx.generator.compute.backend);
}

inline bool uses_scalar_kernel(const action::context & ctx) noexcept {
  return !uses_scalar_packed_q8_0(ctx) &&
      !uses_scalar_q8_k(ctx) &&
      !uses_scalar_native_quantized(ctx);
}

inline bool uses_preselected_scalar_q8_k(const action::context & ctx) noexcept {
  return emel::text::generator::guard::detail::scalar_preselected_argmax_q8_k_supported(
      ctx.generator.compute.backend);
}

inline bool uses_preselected_scalar_kernel(const action::context & ctx) noexcept {
  return !uses_preselected_scalar_q8_k(ctx) && !uses_scalar_native_quantized(ctx);
}

inline bool uses_preselected_scalar_native_quantized_q8_k(const action::context & ctx) noexcept {
  return uses_scalar_native_quantized(ctx) &&
      emel::text::generator::guard::detail::preselected_argmax_output_q8_k_supported(
          ctx.generator.compute.backend);
}

inline bool uses_preselected_scalar_native_quantized_kernel(
    const action::context & ctx) noexcept {
  return uses_scalar_native_quantized(ctx) &&
      !emel::text::generator::guard::detail::preselected_argmax_output_q8_k_supported(
          ctx.generator.compute.backend);
}

inline bool uses_materialized_scalar_native_quantized_q8_k(
    const action::context & ctx) noexcept {
  return uses_scalar_native_quantized(ctx) &&
      emel::text::generator::guard::detail::materialized_output_q8_k_supported(
          ctx.generator.compute.backend);
}

inline bool uses_materialized_scalar_native_quantized_kernel(
    const action::context & ctx) noexcept {
  return uses_scalar_native_quantized(ctx) &&
      !emel::text::generator::guard::detail::materialized_output_q8_k_supported(
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
           emel::text::generator::guard::detail::guard_flash_attention_supported(
               ctx.generator.compute.backend, ev.ctx.prompt_token_count - 1) &&
           emel::text::generator::guard::detail::guard_flash_kv_map_identity(
               ctx.generator, ev.ctx.prompt_token_count);
  }
};

struct nonflash_runtime_required {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !flash_runtime_supported{}(ev, ctx);
  }
};

struct uses_materialized_logits_with_chunk8_q8_k {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !detail::uses_preselected_argmax_direct(ctx) &&
           detail::uses_prefill_chunk8_q8_k_gemm(ev, ctx);
  }
};

struct uses_materialized_logits_with_chunk4_packed_q8_0 {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !detail::uses_preselected_argmax_direct(ctx) &&
           !detail::uses_prefill_chunk8_q8_k_gemm(ev, ctx) &&
           detail::uses_prefill_chunk4_packed_q8_0_gemm(ev, ctx);
  }
};

struct uses_materialized_logits_with_chunk4_q8_k {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !detail::uses_preselected_argmax_direct(ctx) &&
           !detail::uses_prefill_chunk8_q8_k_gemm(ev, ctx) &&
           detail::uses_prefill_chunk4_q8_k_gemm(ev, ctx);
  }
};

struct uses_materialized_logits_with_scalar {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !detail::uses_preselected_argmax_direct(ctx) &&
           !detail::uses_prefill_chunk8_q8_k_gemm(ev, ctx) &&
           !detail::uses_prefill_chunk4_q8_gemm(ev, ctx);
  }
};

struct uses_materialized_logits_with_scalar_packed_q8_0 {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_materialized_logits_with_scalar{}(ev, ctx) && detail::uses_scalar_packed_q8_0(ctx);
  }
};

struct uses_materialized_logits_with_scalar_q8_k {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_materialized_logits_with_scalar{}(ev, ctx) && detail::uses_scalar_q8_k(ctx);
  }
};

struct uses_materialized_logits_with_scalar_kernel {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_materialized_logits_with_scalar{}(ev, ctx) && detail::uses_scalar_kernel(ctx);
  }
};

struct uses_materialized_logits_with_scalar_native_quantized_q8_k {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_materialized_logits_with_scalar{}(ev, ctx) &&
           detail::uses_materialized_scalar_native_quantized_q8_k(ctx);
  }
};

struct uses_materialized_logits_with_scalar_native_quantized_kernel {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_materialized_logits_with_scalar{}(ev, ctx) &&
           detail::uses_materialized_scalar_native_quantized_kernel(ctx);
  }
};

struct uses_preselected_argmax_with_chunk8_q8_k {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_preselected_argmax_direct(ctx) &&
           detail::uses_prefill_chunk8_q8_k_gemm(ev, ctx);
  }
};

struct uses_preselected_argmax_with_chunk4_packed_q8_0 {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_preselected_argmax_direct(ctx) &&
           !detail::uses_prefill_chunk8_q8_k_gemm(ev, ctx) &&
           detail::uses_prefill_chunk4_packed_q8_0_gemm(ev, ctx);
  }
};

struct uses_preselected_argmax_with_chunk4_q8_k {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_preselected_argmax_direct(ctx) &&
           !detail::uses_prefill_chunk8_q8_k_gemm(ev, ctx) &&
           detail::uses_prefill_chunk4_q8_k_gemm(ev, ctx);
  }
};

struct uses_preselected_argmax_with_scalar {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_preselected_argmax_direct(ctx) &&
           !detail::uses_prefill_chunk8_q8_k_gemm(ev, ctx) &&
           !detail::uses_prefill_chunk4_q8_gemm(ev, ctx);
  }
};

struct uses_preselected_argmax_with_scalar_q8_k {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_preselected_argmax_with_scalar{}(ev, ctx) &&
           detail::uses_preselected_scalar_q8_k(ctx);
  }
};

struct uses_preselected_argmax_with_scalar_kernel {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_preselected_argmax_with_scalar{}(ev, ctx) &&
           detail::uses_preselected_scalar_kernel(ctx);
  }
};

struct uses_preselected_argmax_with_scalar_native_quantized_q8_k {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_preselected_argmax_with_scalar{}(ev, ctx) &&
           detail::uses_preselected_scalar_native_quantized_q8_k(ctx);
  }
};

struct uses_preselected_argmax_with_scalar_native_quantized_kernel {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return uses_preselected_argmax_with_scalar{}(ev, ctx) &&
           detail::uses_preselected_scalar_native_quantized_kernel(ctx);
  }
};

struct guard_compute_invalid_request {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return (detail::uses_preselected_argmax_direct(ctx) &&
            emel::text::generator::guard::detail::guard_prefill_preselected_compute_invalid(
                ev.ctx, ctx.generator)) ||
           (!detail::uses_preselected_argmax_direct(ctx) &&
            emel::text::generator::guard::detail::guard_prefill_materialized_compute_invalid(
                ev.ctx, ctx.generator));
  }
};

struct guard_compute_backend_unavailable {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_compute_backend_unavailable(
        ctx.generator);
  }
};

struct guard_materialized_logits_with_chunk8_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_materialized_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_materialized_logits_with_chunk8_q8_k{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_parallel_chunk8_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_parallel_matmul_lanes(ev, ctx) &&
           guard_materialized_logits_with_chunk8_q8_k_ready{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_chunk4_packed_q8_0_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_materialized_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_materialized_logits_with_chunk4_packed_q8_0{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_parallel_matmul_lanes(ev, ctx) &&
           guard_materialized_logits_with_chunk4_packed_q8_0_ready{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_chunk4_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_materialized_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_materialized_logits_with_chunk4_q8_k{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_parallel_chunk4_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_parallel_matmul_lanes(ev, ctx) &&
           guard_materialized_logits_with_chunk4_q8_k_ready{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_scalar_packed_q8_0_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_materialized_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_materialized_logits_with_scalar_packed_q8_0{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_scalar_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_materialized_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_materialized_logits_with_scalar_q8_k{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_scalar_native_quantized_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_materialized_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_materialized_logits_with_scalar_native_quantized_q8_k{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_scalar_native_quantized_kernel_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_materialized_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_materialized_logits_with_scalar_native_quantized_kernel{}(ev, ctx);
  }
};

struct guard_materialized_logits_with_scalar_kernel_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_materialized_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_materialized_logits_with_scalar_kernel{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_chunk8_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_preselected_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_preselected_argmax_with_chunk8_q8_k{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_parallel_chunk8_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_parallel_matmul_lanes(ev, ctx) &&
           guard_preselected_argmax_with_chunk8_q8_k_ready{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_chunk4_packed_q8_0_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_preselected_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_preselected_argmax_with_chunk4_packed_q8_0{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_parallel_matmul_lanes(ev, ctx) &&
           guard_preselected_argmax_with_chunk4_packed_q8_0_ready{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_chunk4_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_preselected_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_preselected_argmax_with_chunk4_q8_k{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_parallel_chunk4_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return detail::uses_parallel_matmul_lanes(ev, ctx) &&
           guard_preselected_argmax_with_chunk4_q8_k_ready{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_scalar_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_preselected_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_preselected_argmax_with_scalar_q8_k{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_scalar_kernel_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_preselected_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_preselected_argmax_with_scalar_kernel{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_preselected_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_preselected_argmax_with_scalar_native_quantized_q8_k{}(ev, ctx);
  }
};

struct guard_preselected_argmax_with_scalar_native_quantized_kernel_ready {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return emel::text::generator::guard::detail::guard_prefill_preselected_compute_ready(
               ev.ctx, ctx.generator) &&
           uses_preselected_argmax_with_scalar_native_quantized_kernel{}(ev, ctx);
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

}  // namespace emel::text::generator::prefill::guard
