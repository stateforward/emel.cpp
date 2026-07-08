#pragma once

#include "emel/text/generator/context.hpp"
#include "emel/text/generator/initializer/context.hpp"
#include "emel/text/generator/initializer/detail.hpp"
#include "emel/graph/errors.hpp"
#include "emel/memory/hybrid/errors.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/generation/any.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/renderer/errors.hpp"

namespace emel::text::generator::initializer::guard {

namespace detail {

inline bool has_phase_success(const event::run & ev) noexcept {
  return ev.ctx.phase_accepted && ev.ctx.phase_code == 0;
}

inline bool phase_rejected_without_code(const event::run & ev) noexcept {
  return !ev.ctx.phase_accepted && ev.ctx.phase_code == 0;
}

constexpr int32_t conditioner_code(const emel::text::conditioner::error err) noexcept {
  return static_cast<int32_t>(err);
}

constexpr int32_t renderer_code(const emel::text::renderer::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t memory_code(const emel::memory::hybrid::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t graph_code(const emel::graph::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

constexpr int32_t loader_code(const emel::model::loader::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

inline bool loader_invalid_code(const int32_t code) noexcept {
  return code == loader_code(emel::model::loader::error::invalid_request) ||
         code == loader_code(emel::model::loader::error::model_invalid);
}

inline bool loader_backend_code(const int32_t code) noexcept {
  return code == loader_code(emel::model::loader::error::backend_error) ||
         code == loader_code(emel::model::loader::error::internal_error) ||
         code == loader_code(emel::model::loader::error::parse_failed) ||
         code == loader_code(emel::model::loader::error::untracked);
}

inline bool conditioner_invalid_code(const int32_t code) noexcept {
  return code == conditioner_code(emel::text::conditioner::error::invalid_argument) ||
         code == conditioner_code(emel::text::conditioner::error::model_invalid) ||
         code == conditioner_code(emel::text::conditioner::error::capacity);
}

inline bool conditioner_backend_code(const int32_t code) noexcept {
  return code == conditioner_code(emel::text::conditioner::error::backend) ||
         code == conditioner_code(emel::text::conditioner::error::untracked);
}

inline bool renderer_invalid_code(const int32_t code) noexcept {
  return code == renderer_code(emel::text::renderer::error::invalid_request) ||
         code == renderer_code(emel::text::renderer::error::model_invalid);
}

inline bool renderer_backend_code(const int32_t code) noexcept {
  return code == renderer_code(emel::text::renderer::error::backend_error) ||
         code == renderer_code(emel::text::renderer::error::internal_error) ||
         code == renderer_code(emel::text::renderer::error::untracked);
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

}  // namespace detail

struct backend_already_ready {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    const auto & backend = ctx.generator.compute.backend;
    const auto & limits = ctx.generator.limits;
    return ctx.generator.compute.backend_ready &&
           limits.block_tokens > 0 &&
           backend.kv_block_tokens == limits.block_tokens &&
           backend.kv_positions_capacity >= backend.n_ctx;
  }
};

struct backend_prepare_needed {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !backend_already_ready{}(ev, ctx);
  }
};

struct guard_generation_contract_valid {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    const auto *contract = ctx.generator.generation_contract;
    if (contract == nullptr) {
      return false;
    }

    const auto *model = contract->execution.model;
    if (model == nullptr || ctx.generator.model != model) {
      return false;
    }

    const int32_t n_head_kv =
        model->params.n_head_kv > 0 ? model->params.n_head_kv : model->params.n_head;
    const int32_t kv_positions_capacity =
        emel::memory::view::positions_capacity_for(ctx.generator.limits.block_tokens,
                                                   model->params.n_ctx);
    return model->params.n_vocab > 0 &&
           model->params.n_embd > 0 &&
           model->params.n_head > 0 &&
           n_head_kv > 0 &&
           model->params.n_ctx > 0 &&
           (model->params.n_embd % model->params.n_head) == 0 &&
           contract->execution.block_count > 0 &&
           contract->generation_execution.layer_count ==
               static_cast<uint32_t>(contract->execution.block_count) &&
           contract->generation_execution.execution == &contract->execution &&
           contract->topology.execution == &contract->execution &&
           contract->topology.node_count > 0u &&
           contract->topology.tensor_count > 0u &&
           contract->prefill_plan.graph == &contract->topology &&
           contract->decode_plan.graph == &contract->topology &&
           contract->prefill_plan.expected_outputs > 0 &&
           contract->decode_plan.expected_outputs > 0 &&
           kv_positions_capacity >= model->params.n_ctx &&
           emel::model::generation::validate_contract(*contract) ==
               emel::error::cast(emel::model::loader::error::none);
  }
};

struct guard_generation_contract_invalid {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !guard_generation_contract_valid{}(ev, ctx);
  }
};

struct guard_backend_reuse_allowed {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return backend_already_ready{}(ev, ctx) &&
           guard_generation_contract_valid{}(ev, ctx);
  }
};

struct guard_backend_prepare_allowed {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return backend_prepare_needed{}(ev, ctx) &&
           guard_generation_contract_valid{}(ev, ctx);
  }
};

struct backend_prepare_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct backend_prepare_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::loader_invalid_code(ev.ctx.phase_code);
  }
};

struct backend_prepare_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::loader_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::loader_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct conditioner_bind_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct conditioner_bind_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::conditioner_invalid_code(ev.ctx.phase_code);
  }
};

struct conditioner_bind_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::conditioner_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::conditioner_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct renderer_initialize_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct renderer_initialize_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::renderer_invalid_code(ev.ctx.phase_code);
  }
};

struct renderer_initialize_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::renderer_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::renderer_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct memory_reserve_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

// Every block id the reserved pool can hand out must map inside the prepared
// physical cache: pool capacity (in tokens) must not exceed the block-padded
// physical position capacity. Under-provisioned pools are valid — outgrowing
// one surfaces as the modeled allocate_slots out_of_memory route.
struct guard_memory_geometry_fits_backend {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    const auto & limits = ctx.generator.limits;
    const auto & backend = ctx.generator.compute.backend;
    const int64_t pool_tokens =
        static_cast<int64_t>(limits.block_capacity) * static_cast<int64_t>(limits.block_tokens);
    return limits.block_tokens > 0 &&
           backend.kv_block_tokens == limits.block_tokens &&
           backend.kv_positions_capacity >= backend.n_ctx &&
           pool_tokens <= static_cast<int64_t>(backend.kv_positions_capacity);
  }
};

struct guard_memory_geometry_gap {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !guard_memory_geometry_fits_backend{}(ev, ctx);
  }
};

struct graph_reservation_present {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    return ctx.generator.state.graph_reservation.node_count > 0u;
  }
};

struct graph_reservation_missing {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return !graph_reservation_present{}(ev, ctx);
  }
};

struct memory_reserve_with_existing_graph {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return memory_reserve_ok{}(ev, ctx) && graph_reservation_present{}(ev, ctx);
  }
};

struct memory_reserve_with_missing_graph {
  bool operator()(const event::run & ev, const action::context & ctx) const noexcept {
    return memory_reserve_ok{}(ev, ctx) && graph_reservation_missing{}(ev, ctx);
  }
};

struct memory_reserve_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::memory_invalid_code(ev.ctx.phase_code);
  }
};

struct memory_reserve_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::memory_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::memory_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct graph_reserve_ok {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return detail::has_phase_success(ev);
  }
};

struct graph_reserve_invalid_request {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !detail::has_phase_success(ev) && detail::graph_invalid_code(ev.ctx.phase_code);
  }
};

struct graph_reserve_backend_error {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    const bool invalid = detail::graph_invalid_code(ev.ctx.phase_code);
    return !detail::has_phase_success(ev) &&
           (detail::phase_rejected_without_code(ev) ||
            detail::graph_backend_code(ev.ctx.phase_code) ||
            !invalid);
  }
};

struct uses_materialized_logits {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    return ctx.generator.state.selection_mode == emel::text::generator::selection_mode::sample_logits;
  }
};

struct uses_preselected_argmax {
  bool operator()(const event::run &, const action::context & ctx) const noexcept {
    return ctx.generator.state.selection_mode ==
           emel::text::generator::selection_mode::preselected_argmax;
  }
};

struct sampler_configured {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return ev.ctx.buffers_ready;
  }
};

struct sampler_config_failed {
  bool operator()(const event::run & ev, const action::context &) const noexcept {
    return !ev.ctx.buffers_ready;
  }
};

}  // namespace emel::text::generator::initializer::guard
