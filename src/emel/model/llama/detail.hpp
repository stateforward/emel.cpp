#pragma once

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/generation/any.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::llama::detail {

inline constexpr uint32_t k_global_tensor_count = 3u;
inline constexpr uint32_t k_block_tensor_count = 8u;
inline constexpr uint32_t k_quantized_stage_family_count =
    emel::model::generation::k_quantized_stage_family_count;

using block_view = emel::model::generation::block_view;
using execution_view = emel::model::generation::execution_view;
using generation_attention_qk_norm_route =
    emel::model::generation::generation_attention_qk_norm_route;
using generation_attention_value_route =
    emel::model::generation::generation_attention_value_route;
using generation_attention_v_norm_route =
    emel::model::generation::generation_attention_v_norm_route;
using generation_attention_window_route =
    emel::model::generation::generation_attention_window_route;
using generation_execution_descriptor =
    emel::model::generation::generation_execution_descriptor;
using generation_layer_execution =
    emel::model::generation::generation_layer_execution;
using generation_residual_route =
    emel::model::generation::generation_residual_route;
using quantized_contract_kind =
    emel::model::generation::quantized_contract_kind;
using quantized_path_audit =
    emel::model::generation::quantized_path_audit;
using quantized_stage_audit =
    emel::model::generation::quantized_stage_audit;
using quantized_stage_family =
    emel::model::generation::quantized_stage_family;
using step_kind = emel::model::generation::step_kind;
using step_plan = emel::model::generation::step_plan;
using tensor_view = emel::model::generation::tensor_view;
using topology = emel::model::generation::topology;

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept;

emel::error::type validate_data(const emel::model::data & model_data) noexcept;

emel::error::type build_generation_contract(
    const emel::model::data & model_data,
    emel::model::generation::contract & contract_out) noexcept;
emel::error::type build_execution_view(const emel::model::data & model_data,
                                       execution_view & view_out) noexcept;
emel::error::type build_generation_execution_descriptor(
    const execution_view & execution,
    generation_execution_descriptor & descriptor_out) noexcept;
emel::error::type build_topology(const execution_view & execution,
                                 topology & topology_out) noexcept;
using emel::model::generation::build_quantized_path_audit;
using emel::model::generation::build_step_plans;
using emel::model::generation::bind_block_tensor_view;
using emel::model::generation::bind_output_view;
using emel::model::generation::bind_tensor_view;
using emel::model::generation::has_tensor_named;
using emel::model::generation::lookup_block_view;
using emel::model::generation::quantized_contract_kind_name;
using emel::model::generation::quantized_stage_family_name;
using emel::model::generation::reject_block_tensor;
using emel::model::generation::require_block_tensor;
using emel::model::generation::tensor_type_name;

}  // namespace emel::model::llama::detail
