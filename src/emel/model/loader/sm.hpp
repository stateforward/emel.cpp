#pragma once

#include "emel/model/loader/actions.hpp"
#include "emel/model/loader/context.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::loader {

struct ready {};
struct request_decision {};
struct parsing {};
struct parse_decision {};
struct parse_phase_decision {};
struct parse_load_tensors_policy_decision {};
struct parse_load_tensors_handler_decision {};
struct loading_tensors {};
struct state_tensor_bind_decision {};
struct state_tensor_plan_dispatch {};
struct state_tensor_plan_decision {};
struct state_io_load_dispatch {};
struct state_io_read_copy_load_dispatch {};
struct state_io_cooperative_async_load_dispatch {};
struct state_io_cooperative_async_resume_dispatch {};
struct state_io_load_decision {};
struct state_io_load_progress_decision {};
struct state_tensor_effect_error_cleanup {};
struct state_tensor_apply_dispatch {};
struct state_tensor_apply_decision {};
struct load_map_policy_decision {};
struct mapping_layers {};
struct map_layers_decision {};
struct structure_decision {};
struct structure_policy_decision {};
struct validating_structure {};
struct structure_validation_decision {};
struct architecture_decision {};
struct architecture_policy_decision {};
struct validating_architecture {};
struct architecture_validation_decision {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<request_decision> <= *sml::state<ready> + sml::event<event::load_runtime>
          / action::begin_load

      , sml::state<state_io_cooperative_async_resume_dispatch> <=
          sml::state<request_decision>
          + sml::completion<event::load_runtime>
          [ guard::cooperative_async_resume_ready{} ]
      , sml::state<parsing> <= sml::state<request_decision>
          + sml::completion<event::load_runtime> [ guard::valid_request{} ]
      , sml::state<errored> <= sml::state<request_decision>
          + sml::completion<event::load_runtime> [ guard::invalid_request{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<parse_decision> <= sml::state<parsing>
          + sml::completion<event::load_runtime> / action::run_parse

      , sml::state<parse_phase_decision> <= sml::state<parse_decision>
          + sml::completion<event::load_runtime>
      , sml::state<parse_load_tensors_policy_decision> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime> [ guard::error_none{} ]
      , sml::state<errored> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime> [ guard::error_invalid_request{} ]
      , sml::state<errored> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime> [ guard::error_parse_failed{} ]
      , sml::state<errored> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime> [ guard::error_backend_error{} ]
      , sml::state<errored> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime> [ guard::error_model_invalid{} ]
      , sml::state<errored> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime> [ guard::error_internal_error{} ]
      , sml::state<errored> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime> [ guard::error_untracked{} ]
      , sml::state<errored> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime>
          [ guard::error_io_strategy_unavailable{} ]
      , sml::state<errored> <= sml::state<parse_phase_decision>
          + sml::completion<event::load_runtime> [ guard::error_unclassified_code{} ]

      , sml::state<parse_load_tensors_handler_decision> <=
          sml::state<parse_load_tensors_policy_decision> + sml::completion<event::load_runtime>
          [ guard::should_load_tensors{} ]
      , sml::state<structure_decision> <= sml::state<parse_load_tensors_policy_decision>
          + sml::completion<event::load_runtime> [ guard::skip_load_tensors{} ]
      , sml::state<errored> <= sml::state<parse_load_tensors_policy_decision>
          + sml::completion<event::load_runtime> / action::mark_internal_error

      , sml::state<loading_tensors> <= sml::state<parse_load_tensors_handler_decision>
          + sml::completion<event::load_runtime> [ guard::can_load_tensors{} ]
      , sml::state<errored> <= sml::state<parse_load_tensors_handler_decision>
          + sml::completion<event::load_runtime>
          [ guard::model_has_no_tensors{} ]
          / action::mark_model_invalid
      , sml::state<errored> <= sml::state<parse_load_tensors_handler_decision>
          + sml::completion<event::load_runtime>
          [ guard::cannot_load_tensors{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<parse_load_tensors_handler_decision>
          + sml::completion<event::load_runtime> / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<state_tensor_bind_decision> <= sml::state<loading_tensors>
          + sml::completion<event::load_runtime>
          / action::effect_dispatch_tensor_bind_storage
      , sml::state<state_tensor_plan_dispatch> <= sml::state<state_tensor_bind_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_bind_done_raised{} ]
      , sml::state<errored> <= sml::state<state_tensor_bind_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_bind_error_raised{} ]
          / action::effect_mark_tensor_bind_error
      , sml::state<errored> <= sml::state<state_tensor_bind_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_bind_unhandled{} ]
          / action::mark_internal_error

      , sml::state<state_tensor_plan_decision> <= sml::state<state_tensor_plan_dispatch>
          + sml::completion<event::load_runtime>
          / action::effect_dispatch_tensor_plan_load
      , sml::state<state_tensor_apply_dispatch> <= sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_without_io_strategy{} ]
      , sml::state<state_tensor_effect_error_cleanup> <=
          sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_io_strategy_without_loader{} ]
          / action::effect_dispatch_tensor_apply_error_results
      , sml::state<state_io_load_dispatch> <= sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_io_strategy_with_loader_and_batch_span_ready{} ]
      , sml::state<state_io_read_copy_load_dispatch> <=
          sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_ready{} ]
      , sml::state<state_io_cooperative_async_load_dispatch> <=
          sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_cooperative_async_storage_ready{} ]
      , sml::state<state_tensor_effect_error_cleanup> <=
          sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing{} ]
          / action::effect_dispatch_tensor_apply_error_results
      , sml::state<state_tensor_effect_error_cleanup> <=
          sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_io_strategy_with_loader_and_batch_span_missing{} ]
          / action::effect_dispatch_tensor_apply_error_results
      , sml::state<errored> <= sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_plan_error_raised{} ]
          / action::effect_mark_tensor_plan_error
      , sml::state<errored> <= sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_plan_unhandled{} ]
          / action::mark_internal_error

      , sml::state<state_io_load_decision> <= sml::state<state_io_load_dispatch>
          + sml::completion<event::load_runtime>
          / action::effect_dispatch_io_load_batch
      , sml::state<state_io_load_decision> <=
          sml::state<state_io_read_copy_load_dispatch>
          + sml::completion<event::load_runtime>
          / action::effect_dispatch_io_read_copy_load_batch
      , sml::state<state_io_load_decision> <=
          sml::state<state_io_cooperative_async_load_dispatch>
          + sml::completion<event::load_runtime>
          / action::effect_dispatch_io_cooperative_async_load_batch
      , sml::state<state_io_load_decision> <=
          sml::state<state_io_cooperative_async_resume_dispatch>
          + sml::completion<event::load_runtime>
          / action::effect_dispatch_io_cooperative_async_resume_batch
      , sml::state<state_tensor_apply_dispatch> <= sml::state<state_io_load_decision>
          + sml::completion<event::load_runtime> [ guard::io_load_done_all{} ]
          / action::effect_mark_io_strategy_used
      , sml::state<state_io_load_progress_decision> <=
          sml::state<state_io_load_decision>
          + sml::completion<event::load_runtime>
          [ guard::io_load_progress_raised{} ]
      , sml::state<state_tensor_effect_error_cleanup> <=
          sml::state<state_io_load_decision>
          + sml::completion<event::load_runtime> [ guard::io_load_error_raised{} ]
          / action::effect_dispatch_tensor_apply_error_results
      , sml::state<state_tensor_effect_error_cleanup> <=
          sml::state<state_io_load_decision>
          + sml::completion<event::load_runtime> [ guard::io_load_unhandled{} ]
          / action::effect_dispatch_tensor_apply_error_results

      , sml::state<ready> <= sml::state<state_io_load_progress_decision>
          + sml::completion<event::load_runtime>
          [ guard::progress_callback_present{} ]
          / action::publish_progress
      , sml::state<ready> <= sml::state<state_io_load_progress_decision>
          + sml::completion<event::load_runtime>
          [ guard::progress_callback_absent{} ]
          / action::publish_progress_noop

      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_io_strategy_without_loader{} ]
          / action::effect_mark_io_strategy_unavailable
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_io_strategy_with_loader_and_batch_span_missing{} ]
          / action::effect_mark_io_strategy_unavailable
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime>
          [ guard::io_load_error_invalid_request{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime>
          [ guard::io_load_error_strategy_unavailable{} ]
          / action::effect_mark_io_strategy_unavailable
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime>
          [ guard::io_load_error_internal{} ]
          / action::mark_internal_error
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime>
          [ guard::io_load_error_untracked{} ]
          / action::mark_untracked
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime>
          [ guard::io_load_error_unclassified{} ]
          / action::mark_internal_error
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime> [ guard::io_load_unhandled{} ]
          / action::mark_internal_error
      , sml::state<errored> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::completion<event::load_runtime> / action::mark_internal_error

      , sml::state<state_tensor_apply_decision> <= sml::state<state_tensor_apply_dispatch>
          + sml::completion<event::load_runtime>
          / action::effect_dispatch_tensor_apply_results
      , sml::state<load_map_policy_decision> <= sml::state<state_tensor_apply_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_apply_done_with_file_image{} ]
          / action::effect_publish_tensor_load_done_from_file_image
      , sml::state<load_map_policy_decision> <= sml::state<state_tensor_apply_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_apply_done_without_file_image{} ]
          / action::effect_publish_tensor_load_done_from_model_data
      , sml::state<errored> <= sml::state<state_tensor_apply_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_apply_error_raised{} ]
          / action::effect_mark_tensor_apply_error
      , sml::state<errored> <= sml::state<state_tensor_apply_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_apply_unhandled{} ]
          / action::mark_internal_error

      , sml::state<mapping_layers> <= sml::state<load_map_policy_decision>
          + sml::completion<event::load_runtime> [ guard::can_map_layers{} ]
      , sml::state<errored> <= sml::state<load_map_policy_decision>
          + sml::completion<event::load_runtime> [ guard::cannot_map_layers{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<load_map_policy_decision>
          + sml::completion<event::load_runtime> / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<map_layers_decision> <= sml::state<mapping_layers>
          + sml::completion<event::load_runtime> / action::run_map_layers

      , sml::state<structure_decision> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::error_none{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::error_invalid_request{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::error_parse_failed{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::error_backend_error{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::error_model_invalid{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::error_internal_error{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::error_untracked{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime>
          [ guard::error_io_strategy_unavailable{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::error_unclassified_code{} ]

      //------------------------------------------------------------------------------//
      , sml::state<structure_policy_decision> <= sml::state<structure_decision>
          + sml::completion<event::load_runtime>
      , sml::state<architecture_decision> <= sml::state<structure_policy_decision>
          + sml::completion<event::load_runtime> [ guard::skip_validate_structure{} ]
      , sml::state<validating_structure> <= sml::state<structure_policy_decision>
          + sml::completion<event::load_runtime> [ guard::can_validate_structure{} ]
      , sml::state<errored> <= sml::state<structure_policy_decision>
          + sml::completion<event::load_runtime> [ guard::cannot_validate_structure{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<structure_policy_decision>
          + sml::completion<event::load_runtime> / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<structure_validation_decision> <= sml::state<validating_structure>
          + sml::completion<event::load_runtime> / action::run_validate_structure

      , sml::state<architecture_decision> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_none{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_invalid_request{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_parse_failed{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_backend_error{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_model_invalid{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_internal_error{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_untracked{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime>
          [ guard::error_io_strategy_unavailable{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_unclassified_code{} ]

      //------------------------------------------------------------------------------//
      , sml::state<architecture_policy_decision> <= sml::state<architecture_decision>
          + sml::completion<event::load_runtime>
      , sml::state<done> <= sml::state<architecture_policy_decision>
          + sml::completion<event::load_runtime> [ guard::skip_validate_architecture{} ]
      , sml::state<validating_architecture> <= sml::state<architecture_policy_decision>
          + sml::completion<event::load_runtime> [ guard::can_validate_architecture{} ]
      , sml::state<errored> <= sml::state<architecture_policy_decision>
          + sml::completion<event::load_runtime>
          [ guard::cannot_validate_architecture{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<architecture_policy_decision>
          + sml::completion<event::load_runtime> / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<architecture_validation_decision> <= sml::state<validating_architecture>
          + sml::completion<event::load_runtime> / action::run_validate_architecture

      , sml::state<done> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_none{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_invalid_request{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_parse_failed{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_backend_error{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_model_invalid{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_internal_error{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_untracked{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime>
          [ guard::error_io_strategy_unavailable{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::error_unclassified_code{} ]

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<event::load_runtime>
          [ guard::done_callback_present{} ]
          / action::publish_done
      , sml::state<ready> <= sml::state<done> + sml::completion<event::load_runtime>
          [ guard::done_callback_absent{} ]
          / action::publish_done_noop
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::load_runtime>
          [ guard::error_callback_present{} ]
          / action::publish_error
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::load_runtime>
          [ guard::error_callback_absent{} ]
          / action::publish_error_noop

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<parsing> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<parse_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<parse_phase_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<parse_load_tensors_policy_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<parse_load_tensors_handler_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<loading_tensors> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<state_tensor_bind_decision>
          + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<state_tensor_plan_dispatch>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_tensor_plan_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_io_load_dispatch>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_io_read_copy_load_dispatch>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <=
          sml::state<state_io_cooperative_async_load_dispatch>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <=
          sml::state<state_io_cooperative_async_resume_dispatch>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_io_load_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_io_load_progress_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_tensor_effect_error_cleanup>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_tensor_apply_dispatch>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_tensor_apply_decision>
          + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<load_map_policy_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<mapping_layers> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<map_layers_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<structure_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<structure_policy_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<validating_structure> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<structure_validation_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<architecture_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<architecture_policy_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<validating_architecture> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<architecture_validation_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<errored> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  sm() : base_type() {}

  bool process_event(const event::load &ev) {
    event::load_ctx ctx{};
    events::tensor_bind_done tensor_bind_done{};
    events::tensor_bind_error tensor_bind_error{};
    events::tensor_plan_done tensor_plan_done{};
    events::tensor_plan_error tensor_plan_error{};
    events::tensor_apply_done tensor_apply_done{};
    events::tensor_apply_error tensor_apply_error{};
    events::io_load_done io_load_done{};
    events::io_load_progress io_load_progress{};
    events::io_load_error io_load_error{};
    event::io_phase_events io_events{
        .load_done = io_load_done,
        .load_progress = io_load_progress,
        .load_error = io_load_error,
    };
    event::load_runtime runtime{
        ev,
        ctx,
        event::tensor_phase_events{
            .bind_done = tensor_bind_done,
            .bind_error = tensor_bind_error,
            .plan_done = tensor_plan_done,
            .plan_error = tensor_plan_error,
            .apply_done = tensor_apply_done,
            .apply_error = tensor_apply_error,
        },
        &io_events,
    };
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

} // namespace emel::model::loader
