#pragma once

#include "emel/generator/actions.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/prefill/context.hpp"
#include "emel/generator/prefill/detail.hpp"

namespace emel::generator::prefill::action {

namespace detail {

struct request_slots {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    ev.ctx.phase_code = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    emel::memory::event::allocate_slots allocate_ev{
      .seq_id = emel::generator::action::k_sequence_id,
      .token_count = ev.ctx.prompt_token_count,
      .block_count_out = nullptr,
      .error_out = &ev.ctx.phase_code,
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

template <emel::generator::prefill_compute_contract contract, auto run_kernel_fn>
inline void request_compute_contract(const event::run & ev, context & ctx) noexcept {
  ev.ctx.prefill_contract = contract;
  const emel::generator::event::generate_run runtime{ev.request, ev.ctx};
  emel::generator::action::request_phase_compute<
      emel::generator::detail::step_kind::prefill,
      run_kernel_fn>(runtime, ctx.generator);
}

template <emel::generator::prefill_compute_contract contract, auto run_kernel_fn>
inline void request_compute_contract_preselected_argmax(const event::run & ev,
                                                        context & ctx) noexcept {
  ev.ctx.prefill_contract = contract;
  const emel::generator::event::generate_run runtime{ev.request, ev.ctx};
  emel::generator::action::request_phase_compute_preselected_argmax<
      emel::generator::detail::step_kind::prefill,
      run_kernel_fn>(runtime, ctx.generator);
}

struct mark_prefill_cached {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.ctx.kv_tokens = ev.ctx.prompt_token_count;
    ev.ctx.err = emel::error::cast(emel::generator::error::none);
  }
};

struct mark_invalid_request {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(emel::generator::error::invalid_request);
  }
};

struct mark_backend_error {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(emel::generator::error::backend);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(emel::generator::error::backend);
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

struct request_contract_flash_materialized_scalar {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::generator::prefill_compute_contract::flash_materialized_scalar,
        emel::generator::detail::run_kernel_flash>(ev, ctx);
  }
};

struct request_contract_flash_materialized_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::generator::prefill_compute_contract::flash_materialized_chunk4_packed_q8_0,
        emel::generator::detail::run_kernel_flash_prefill_chunk4_packed_q8_0>(ev, ctx);
  }
};

struct request_contract_flash_materialized_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::generator::prefill_compute_contract::flash_materialized_chunk4_q8_k,
        emel::generator::detail::run_kernel_flash_prefill_chunk4_q8_k>(ev, ctx);
  }
};

struct request_contract_flash_preselected_scalar {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::generator::prefill_compute_contract::flash_preselected_scalar,
        emel::generator::detail::run_kernel_flash_preselected_argmax>(ev, ctx);
  }
};

struct request_contract_flash_preselected_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::generator::prefill_compute_contract::flash_preselected_chunk4_packed_q8_0,
        emel::generator::detail::run_kernel_flash_prefill_chunk4_preselected_argmax_packed_q8_0>(
        ev, ctx);
  }
};

struct request_contract_flash_preselected_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::generator::prefill_compute_contract::flash_preselected_chunk4_q8_k,
        emel::generator::detail::run_kernel_flash_prefill_chunk4_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

struct request_contract_nonflash_materialized_scalar {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::generator::prefill_compute_contract::nonflash_materialized_scalar,
        emel::generator::detail::run_kernel_nonflash>(ev, ctx);
  }
};

struct request_contract_nonflash_materialized_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::generator::prefill_compute_contract::nonflash_materialized_chunk4_packed_q8_0,
        emel::generator::detail::run_kernel_nonflash_prefill_chunk4_packed_q8_0>(ev, ctx);
  }
};

struct request_contract_nonflash_materialized_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract<
        emel::generator::prefill_compute_contract::nonflash_materialized_chunk4_q8_k,
        emel::generator::detail::run_kernel_nonflash_prefill_chunk4_q8_k>(ev, ctx);
  }
};

struct request_contract_nonflash_preselected_scalar {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::generator::prefill_compute_contract::nonflash_preselected_scalar,
        emel::generator::detail::run_kernel_nonflash_preselected_argmax>(ev, ctx);
  }
};

struct request_contract_nonflash_preselected_chunk4_packed_q8_0 {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::generator::prefill_compute_contract::nonflash_preselected_chunk4_packed_q8_0,
        emel::generator::detail::run_kernel_nonflash_prefill_chunk4_preselected_argmax_packed_q8_0>(
        ev, ctx);
  }
};

struct request_contract_nonflash_preselected_chunk4_q8_k {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    detail::request_compute_contract_preselected_argmax<
        emel::generator::prefill_compute_contract::nonflash_preselected_chunk4_q8_k,
        emel::generator::detail::run_kernel_nonflash_prefill_chunk4_preselected_argmax_q8_k>(
        ev, ctx);
  }
};

inline constexpr request_slots request_slots{};
inline constexpr request_memory_snapshot request_memory_snapshot{};
inline constexpr request_contract_flash_materialized_scalar
    request_contract_flash_materialized_scalar{};
inline constexpr request_contract_flash_materialized_chunk4_packed_q8_0
    request_contract_flash_materialized_chunk4_packed_q8_0{};
inline constexpr request_contract_flash_materialized_chunk4_q8_k
    request_contract_flash_materialized_chunk4_q8_k{};
inline constexpr request_contract_flash_preselected_scalar
    request_contract_flash_preselected_scalar{};
inline constexpr request_contract_flash_preselected_chunk4_packed_q8_0
    request_contract_flash_preselected_chunk4_packed_q8_0{};
inline constexpr request_contract_flash_preselected_chunk4_q8_k
    request_contract_flash_preselected_chunk4_q8_k{};
inline constexpr request_contract_nonflash_materialized_scalar
    request_contract_nonflash_materialized_scalar{};
inline constexpr request_contract_nonflash_materialized_chunk4_packed_q8_0
    request_contract_nonflash_materialized_chunk4_packed_q8_0{};
inline constexpr request_contract_nonflash_materialized_chunk4_q8_k
    request_contract_nonflash_materialized_chunk4_q8_k{};
inline constexpr request_contract_nonflash_preselected_scalar
    request_contract_nonflash_preselected_scalar{};
inline constexpr request_contract_nonflash_preselected_chunk4_packed_q8_0
    request_contract_nonflash_preselected_chunk4_packed_q8_0{};
inline constexpr request_contract_nonflash_preselected_chunk4_q8_k
    request_contract_nonflash_preselected_chunk4_q8_k{};
inline constexpr mark_prefill_cached mark_prefill_cached{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_backend_error mark_backend_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::generator::prefill::action
