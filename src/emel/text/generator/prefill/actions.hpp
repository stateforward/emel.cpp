#pragma once

#include "emel/text/generator/actions.hpp"
#include "emel/text/generator/errors.hpp"
#include "emel/text/generator/prefill/context.hpp"
#include "emel/text/generator/prefill/detail.hpp"

namespace emel::text::generator::prefill::action {

namespace detail {

struct request_slots {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::allocate_slots allocate_ev{
      .seq_id = emel::text::generator::action::k_sequence_id,
      .token_count = ev.ctx.prompt_token_count,
      .block_count_out = nullptr,
      .error_out = &ev.ctx.phase_code,
      .copy_block = emel::text::generator::action::copy_kv_cache_block,
      .copy_block_user_data = &ctx.generator.compute.backend,
    };
    ev.ctx.phase_accepted = ctx.generator.memory.process_event(allocate_ev);
  }
};

struct request_memory_snapshot {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::capture_view capture_ev{
      .snapshot_out = &ctx.generator.state.memory_snapshot,
      .error_out = &ev.ctx.phase_code,
    };
    ev.ctx.phase_accepted = ctx.generator.memory.process_event(capture_ev);
  }
};

template <emel::text::generator::prefill_compute_contract contract, auto run_kernel_fn>
inline void request_compute_contract(const event::run & ev, context & ctx) noexcept {
  ev.ctx.prefill_contract = contract;
  const emel::text::generator::event::generate_run runtime{ev.request, ev.ctx};
  emel::text::generator::action::request_phase_compute<
      emel::text::generator::detail::step_kind::prefill,
      run_kernel_fn>(runtime, ctx.generator);
}

template <emel::text::generator::prefill_compute_contract contract, auto run_kernel_fn>
inline void request_compute_contract_preselected_argmax(const event::run & ev,
                                                        context & ctx) noexcept {
  ev.ctx.prefill_contract = contract;
  const emel::text::generator::event::generate_run runtime{ev.request, ev.ctx};
  emel::text::generator::action::request_phase_compute_preselected_argmax<
      emel::text::generator::detail::step_kind::prefill,
      run_kernel_fn>(runtime, ctx.generator);
}

struct mark_prefill_cached {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.ctx.kv_tokens = ev.ctx.prompt_token_count;
    ev.ctx.err = emel::error::cast(emel::text::generator::error::none);
  }
};

struct mark_invalid_request {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(emel::text::generator::error::invalid_request);
  }
};

struct mark_backend_error {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(emel::text::generator::error::backend);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(emel::text::generator::error::backend);
    }
  }
};

}  // namespace detail

using detail::mark_backend_error;
using detail::mark_invalid_request;
using detail::mark_prefill_cached;
using detail::on_unexpected;
using detail::request_memory_snapshot;
using detail::request_slots;

struct request_contract_flash_materialized_scalar_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_scalar,
        emel::text::generator::detail::run_kernel_flash_prefill_scalar_packed_q8_0>(ev, ctx);
  }
};

struct request_contract_flash_materialized_scalar_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_scalar,
        emel::text::generator::detail::run_kernel_flash_prefill_scalar_q8_k>(ev, ctx);
  }
};

struct request_contract_flash_materialized_scalar_native_quantized {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_scalar,
        emel::text::generator::detail::run_kernel_flash_prefill_scalar_native_quantized>(ev, ctx);
  }
};

struct request_contract_flash_materialized_scalar_native_quantized_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_scalar,
        emel::text::generator::detail::
            run_kernel_flash_prefill_scalar_native_quantized_q8_k_logits>(ev, ctx);
  }
};

struct request_contract_flash_materialized_scalar_kernel {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_scalar,
        emel::text::generator::detail::run_kernel_flash_prefill_scalar_kernel>(ev, ctx);
  }
};

struct request_contract_flash_materialized_chunk8_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_chunk8_q8_k,
        emel::text::generator::detail::run_kernel_flash_prefill_chunk8_q8_k>(ev, ctx);
  }
};

struct request_contract_flash_materialized_parallel_chunk8_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_chunk8_q8_k,
        emel::text::generator::detail::run_kernel_flash_prefill_parallel_chunk8_q8_k>(ev, ctx);
  }
};

struct request_contract_flash_materialized_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_chunk4_packed_q8_0,
        emel::text::generator::detail::run_kernel_flash_prefill_chunk4_packed_q8_0>(ev, ctx);
  }
};

struct request_contract_flash_materialized_parallel_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_chunk4_packed_q8_0,
        emel::text::generator::detail::run_kernel_flash_prefill_parallel_chunk4_packed_q8_0>(
        ev, ctx);
  }
};

struct request_contract_flash_materialized_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_chunk4_q8_k,
        emel::text::generator::detail::run_kernel_flash_prefill_chunk4_q8_k>(ev, ctx);
  }
};

struct request_contract_flash_materialized_parallel_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::flash_materialized_chunk4_q8_k,
        emel::text::generator::detail::run_kernel_flash_prefill_parallel_chunk4_q8_k>(ev, ctx);
  }
};

struct request_contract_flash_preselected_scalar_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_scalar,
        emel::text::generator::detail::run_kernel_flash_prefill_scalar_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_scalar_native_quantized_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_scalar,
        emel::text::generator::detail::
            run_kernel_flash_prefill_scalar_preselected_argmax_native_quantized_q8_k>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_scalar_native_quantized_kernel {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_scalar,
        emel::text::generator::detail::
            run_kernel_flash_prefill_scalar_preselected_argmax_native_quantized_kernel>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_scalar_kernel {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_scalar,
        emel::text::generator::detail::run_kernel_flash_prefill_scalar_preselected_argmax_kernel>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_chunk8_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_chunk8_q8_k,
        emel::text::generator::detail::run_kernel_flash_prefill_chunk8_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_parallel_chunk8_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_chunk8_q8_k,
        emel::text::generator::detail::
            run_kernel_flash_prefill_parallel_chunk8_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_chunk4_packed_q8_0,
        emel::text::generator::detail::run_kernel_flash_prefill_chunk4_preselected_argmax_packed_q8_0>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_parallel_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_chunk4_packed_q8_0,
        emel::text::generator::detail::
            run_kernel_flash_prefill_parallel_chunk4_preselected_argmax_packed_q8_0>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_chunk4_q8_k,
        emel::text::generator::detail::run_kernel_flash_prefill_chunk4_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_parallel_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::flash_preselected_chunk4_q8_k,
        emel::text::generator::detail::
            run_kernel_flash_prefill_parallel_chunk4_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

struct request_contract_nonflash_materialized_scalar_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::nonflash_materialized_scalar,
        emel::text::generator::detail::run_kernel_nonflash_prefill_scalar_packed_q8_0>(ev, ctx);
  }
};

struct request_contract_nonflash_materialized_scalar_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::nonflash_materialized_scalar,
        emel::text::generator::detail::run_kernel_nonflash_prefill_scalar_q8_k>(ev, ctx);
  }
};

struct request_contract_nonflash_materialized_scalar_native_quantized {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::nonflash_materialized_scalar,
        emel::text::generator::detail::run_kernel_nonflash_prefill_scalar_native_quantized>(
        ev, ctx);
  }
};

struct request_contract_nonflash_materialized_scalar_native_quantized_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::nonflash_materialized_scalar,
        emel::text::generator::detail::
            run_kernel_nonflash_prefill_scalar_native_quantized_q8_k_logits>(ev, ctx);
  }
};

struct request_contract_nonflash_materialized_scalar_kernel {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::nonflash_materialized_scalar,
        emel::text::generator::detail::run_kernel_nonflash_prefill_scalar_kernel>(ev, ctx);
  }
};

struct request_contract_nonflash_materialized_chunk8_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::nonflash_materialized_chunk8_q8_k,
        emel::text::generator::detail::run_kernel_nonflash_prefill_chunk8_q8_k>(ev, ctx);
  }
};

struct request_contract_nonflash_materialized_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::nonflash_materialized_chunk4_packed_q8_0,
        emel::text::generator::detail::run_kernel_nonflash_prefill_chunk4_packed_q8_0>(ev, ctx);
  }
};

struct request_contract_nonflash_materialized_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::text::generator::prefill_compute_contract::nonflash_materialized_chunk4_q8_k,
        emel::text::generator::detail::run_kernel_nonflash_prefill_chunk4_q8_k>(ev, ctx);
  }
};

struct request_contract_nonflash_preselected_scalar_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::nonflash_preselected_scalar,
        emel::text::generator::detail::
            run_kernel_nonflash_prefill_scalar_preselected_argmax_q8_k>(ev, ctx);
  }
};

struct request_contract_nonflash_preselected_scalar_native_quantized_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::nonflash_preselected_scalar,
        emel::text::generator::detail::
            run_kernel_nonflash_prefill_scalar_preselected_argmax_native_quantized_q8_k>(
        ev, ctx);
  }
};

struct request_contract_nonflash_preselected_scalar_native_quantized_kernel {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::nonflash_preselected_scalar,
        emel::text::generator::detail::
            run_kernel_nonflash_prefill_scalar_preselected_argmax_native_quantized_kernel>(
        ev, ctx);
  }
};

struct request_contract_nonflash_preselected_scalar_kernel {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::nonflash_preselected_scalar,
        emel::text::generator::detail::
            run_kernel_nonflash_prefill_scalar_preselected_argmax_kernel>(ev, ctx);
  }
};

struct request_contract_nonflash_preselected_chunk8_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::nonflash_preselected_chunk8_q8_k,
        emel::text::generator::detail::run_kernel_nonflash_prefill_chunk8_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

struct request_contract_nonflash_preselected_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::nonflash_preselected_chunk4_packed_q8_0,
        emel::text::generator::detail::run_kernel_nonflash_prefill_chunk4_preselected_argmax_packed_q8_0>(
        ev, ctx);
  }
};

struct request_contract_nonflash_preselected_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::text::generator::prefill_compute_contract::nonflash_preselected_chunk4_q8_k,
        emel::text::generator::detail::run_kernel_nonflash_prefill_chunk4_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

inline constexpr request_slots request_slots{};
inline constexpr request_memory_snapshot request_memory_snapshot{};
inline constexpr request_contract_flash_materialized_scalar_packed_q8_0
    request_contract_flash_materialized_scalar_packed_q8_0{};
inline constexpr request_contract_flash_materialized_scalar_q8_k
    request_contract_flash_materialized_scalar_q8_k{};
inline constexpr request_contract_flash_materialized_scalar_native_quantized
    request_contract_flash_materialized_scalar_native_quantized{};
inline constexpr request_contract_flash_materialized_scalar_native_quantized_q8_k
    request_contract_flash_materialized_scalar_native_quantized_q8_k{};
inline constexpr request_contract_flash_materialized_scalar_kernel
    request_contract_flash_materialized_scalar_kernel{};
inline constexpr request_contract_flash_materialized_chunk8_q8_k
    request_contract_flash_materialized_chunk8_q8_k{};
inline constexpr request_contract_flash_materialized_parallel_chunk8_q8_k
    request_contract_flash_materialized_parallel_chunk8_q8_k{};
inline constexpr request_contract_flash_materialized_chunk4_packed_q8_0
    request_contract_flash_materialized_chunk4_packed_q8_0{};
inline constexpr request_contract_flash_materialized_parallel_chunk4_packed_q8_0
    request_contract_flash_materialized_parallel_chunk4_packed_q8_0{};
inline constexpr request_contract_flash_materialized_chunk4_q8_k
    request_contract_flash_materialized_chunk4_q8_k{};
inline constexpr request_contract_flash_materialized_parallel_chunk4_q8_k
    request_contract_flash_materialized_parallel_chunk4_q8_k{};
inline constexpr request_contract_flash_preselected_scalar_q8_k
    request_contract_flash_preselected_scalar_q8_k{};
inline constexpr request_contract_flash_preselected_scalar_native_quantized_q8_k
    request_contract_flash_preselected_scalar_native_quantized_q8_k{};
inline constexpr request_contract_flash_preselected_scalar_native_quantized_kernel
    request_contract_flash_preselected_scalar_native_quantized_kernel{};
inline constexpr request_contract_flash_preselected_scalar_kernel
    request_contract_flash_preselected_scalar_kernel{};
inline constexpr request_contract_flash_preselected_chunk8_q8_k
    request_contract_flash_preselected_chunk8_q8_k{};
inline constexpr request_contract_flash_preselected_parallel_chunk8_q8_k
    request_contract_flash_preselected_parallel_chunk8_q8_k{};
inline constexpr request_contract_flash_preselected_chunk4_packed_q8_0
    request_contract_flash_preselected_chunk4_packed_q8_0{};
inline constexpr request_contract_flash_preselected_parallel_chunk4_packed_q8_0
    request_contract_flash_preselected_parallel_chunk4_packed_q8_0{};
inline constexpr request_contract_flash_preselected_chunk4_q8_k
    request_contract_flash_preselected_chunk4_q8_k{};
inline constexpr request_contract_flash_preselected_parallel_chunk4_q8_k
    request_contract_flash_preselected_parallel_chunk4_q8_k{};
inline constexpr request_contract_nonflash_materialized_scalar_packed_q8_0
    request_contract_nonflash_materialized_scalar_packed_q8_0{};
inline constexpr request_contract_nonflash_materialized_scalar_q8_k
    request_contract_nonflash_materialized_scalar_q8_k{};
inline constexpr request_contract_nonflash_materialized_scalar_native_quantized
    request_contract_nonflash_materialized_scalar_native_quantized{};
inline constexpr request_contract_nonflash_materialized_scalar_native_quantized_q8_k
    request_contract_nonflash_materialized_scalar_native_quantized_q8_k{};
inline constexpr request_contract_nonflash_materialized_scalar_kernel
    request_contract_nonflash_materialized_scalar_kernel{};
inline constexpr request_contract_nonflash_materialized_chunk8_q8_k
    request_contract_nonflash_materialized_chunk8_q8_k{};
inline constexpr request_contract_nonflash_materialized_chunk4_packed_q8_0
    request_contract_nonflash_materialized_chunk4_packed_q8_0{};
inline constexpr request_contract_nonflash_materialized_chunk4_q8_k
    request_contract_nonflash_materialized_chunk4_q8_k{};
inline constexpr request_contract_nonflash_preselected_scalar_q8_k
    request_contract_nonflash_preselected_scalar_q8_k{};
inline constexpr request_contract_nonflash_preselected_scalar_native_quantized_q8_k
    request_contract_nonflash_preselected_scalar_native_quantized_q8_k{};
inline constexpr request_contract_nonflash_preselected_scalar_native_quantized_kernel
    request_contract_nonflash_preselected_scalar_native_quantized_kernel{};
inline constexpr request_contract_nonflash_preselected_scalar_kernel
    request_contract_nonflash_preselected_scalar_kernel{};
inline constexpr request_contract_nonflash_preselected_chunk8_q8_k
    request_contract_nonflash_preselected_chunk8_q8_k{};
inline constexpr request_contract_nonflash_preselected_chunk4_packed_q8_0
    request_contract_nonflash_preselected_chunk4_packed_q8_0{};
inline constexpr request_contract_nonflash_preselected_chunk4_q8_k
    request_contract_nonflash_preselected_chunk4_q8_k{};
inline constexpr mark_prefill_cached mark_prefill_cached{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::generator::prefill::action
