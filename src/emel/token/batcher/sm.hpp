#pragma once

#include "emel/sm.hpp"
#include "emel/token/batcher/actions.hpp"
#include "emel/token/batcher/errors.hpp"
#include "emel/token/batcher/events.hpp"
#include "emel/token/batcher/guards.hpp"

namespace emel::token::batcher {

struct ready {};
struct request_decision {};
struct request_validation_probe {};
struct request_outputs_decision {};
struct request_token_counts_decision {};
struct request_capacities_decision {};
struct request_token_ids_decision {};
struct request_seq_payload_decision {};
struct seq_mode_decision {};
struct seq_from_masks {};
struct seq_from_primary_ids {};
struct seq_default {};
struct seq_mask_words_publish_decision {};
struct positions_mode_decision {};
struct positions_copy_stride_three {};
struct positions_copy_stride_one {};
struct positions_seeded_probe {};
struct positions_unseeded_probe {};
struct positions_generate_seeded {};
struct positions_generate_unseeded {};
struct positions_count_publish_decision {};
struct output_mode_decision {};
struct output_mask_all {};
struct output_mask_copy {};
struct output_mask_last {};
struct output_counting {};
struct outputs_total_publish_decision {};
struct single_output_decision {};
struct single_output_probe {};
struct continuity_decision {};
struct continuity_probe {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<request_decision> <= *sml::state<ready> + sml::event<event::batch_runtime>
          / action::begin_batch

      , sml::state<request_validation_probe> <= sml::state<request_decision>
          + sml::completion<event::batch_runtime>
      , sml::state<request_outputs_decision> <= sml::state<request_validation_probe>
          + sml::completion<event::batch_runtime>
      , sml::state<request_token_counts_decision> <= sml::state<request_outputs_decision>
          + sml::completion<event::batch_runtime> [ guard::request_outputs_present{} ]
      , sml::state<errored> <= sml::state<request_outputs_decision>
          + sml::completion<event::batch_runtime> [ guard::request_outputs_missing{} ]
          / action::mark_invalid_request
      , sml::state<request_capacities_decision> <= sml::state<request_token_counts_decision>
          + sml::completion<event::batch_runtime> [ guard::request_token_counts_valid{} ]
      , sml::state<errored> <= sml::state<request_token_counts_decision>
          + sml::completion<event::batch_runtime> [ guard::request_token_counts_invalid{} ]
          / action::mark_invalid_request
      , sml::state<request_token_ids_decision> <= sml::state<request_capacities_decision>
          + sml::completion<event::batch_runtime> [ guard::request_capacities_valid{} ]
      , sml::state<errored> <= sml::state<request_capacities_decision>
          + sml::completion<event::batch_runtime> [ guard::request_capacities_invalid{} ]
          / action::mark_invalid_request
      , sml::state<request_seq_payload_decision> <= sml::state<request_token_ids_decision>
          + sml::completion<event::batch_runtime> [ guard::request_token_ids_in_vocab{} ]
      , sml::state<errored> <= sml::state<request_token_ids_decision>
          + sml::completion<event::batch_runtime> [ guard::request_token_ids_out_of_vocab{} ]
          / action::mark_invalid_request
      , sml::state<seq_mode_decision> <= sml::state<request_seq_payload_decision>
          + sml::completion<event::batch_runtime> [ guard::request_seq_payload_valid{} ]
      , sml::state<errored> <= sml::state<request_seq_payload_decision>
          + sml::completion<event::batch_runtime> [ guard::request_seq_payload_invalid{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<seq_from_masks> <= sml::state<seq_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::seq_mode_masks{} ]
          / action::normalize_seq_from_masks
      , sml::state<seq_from_primary_ids> <= sml::state<seq_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::seq_mode_primary_ids{} ]
          / action::normalize_seq_from_primary_ids
      , sml::state<seq_default> <= sml::state<seq_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::seq_mode_default{} ]
          / action::normalize_seq_default
      , sml::state<errored> <= sml::state<seq_mode_decision>
          + sml::completion<event::batch_runtime>
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<seq_mask_words_publish_decision> <= sml::state<seq_from_masks>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
      , sml::state<errored> <= sml::state<seq_from_masks>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<seq_from_masks>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<seq_from_masks>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<seq_from_masks>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]
      , sml::state<seq_mask_words_publish_decision> <= sml::state<seq_from_primary_ids>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
      , sml::state<errored> <= sml::state<seq_from_primary_ids>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<seq_from_primary_ids>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<seq_from_primary_ids>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<seq_from_primary_ids>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]
      , sml::state<seq_mask_words_publish_decision> <= sml::state<seq_default>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
      , sml::state<errored> <= sml::state<seq_default>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<seq_default>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<seq_default>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<seq_default>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]

      //------------------------------------------------------------------------------//
      , sml::state<positions_mode_decision> <= sml::state<seq_mask_words_publish_decision>
          + sml::completion<event::batch_runtime> [ guard::seq_mask_words_out_present{} ]
          / action::publish_seq_mask_words
      , sml::state<positions_mode_decision> <= sml::state<seq_mask_words_publish_decision>
          + sml::completion<event::batch_runtime> [ guard::seq_mask_words_out_absent{} ]

      //------------------------------------------------------------------------------//
      , sml::state<positions_copy_stride_three> <= sml::state<positions_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::positions_mode_stride_three{} ]
          / action::copy_positions_stride_three
      , sml::state<positions_copy_stride_one> <= sml::state<positions_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::positions_mode_stride_one{} ]
          / action::copy_positions_stride_one
      , sml::state<positions_seeded_probe> <= sml::state<positions_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::positions_mode_generate_seeded{} ]
          / action::probe_positions_seeded
      , sml::state<positions_unseeded_probe> <= sml::state<positions_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::positions_mode_generate_unseeded{} ]
          / action::probe_positions_unseeded
      , sml::state<errored> <= sml::state<positions_mode_decision>
          + sml::completion<event::batch_runtime>
          / action::mark_internal_error

      , sml::state<positions_generate_seeded> <= sml::state<positions_seeded_probe>
          + sml::completion<event::batch_runtime> [ guard::positions_seeded_probe_ok{} ]
          / action::generate_positions_seeded
      , sml::state<errored> <= sml::state<positions_seeded_probe>
          + sml::completion<event::batch_runtime>
          [ guard::positions_seeded_probe_backend_error{} ]
          / action::mark_backend_error
      , sml::state<errored> <= sml::state<positions_seeded_probe>
          + sml::completion<event::batch_runtime>
          [ guard::positions_seeded_probe_invalid_request{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<positions_seeded_probe>
          + sml::completion<event::batch_runtime>
          / action::mark_internal_error

      , sml::state<positions_generate_unseeded> <= sml::state<positions_unseeded_probe>
          + sml::completion<event::batch_runtime> [ guard::positions_unseeded_probe_ok{} ]
          / action::generate_positions_unseeded
      , sml::state<errored> <= sml::state<positions_unseeded_probe>
          + sml::completion<event::batch_runtime>
          [ guard::positions_unseeded_probe_invalid_request{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<positions_unseeded_probe>
          + sml::completion<event::batch_runtime>
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<positions_count_publish_decision> <= sml::state<positions_copy_stride_three>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
      , sml::state<errored> <= sml::state<positions_copy_stride_three>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<positions_copy_stride_three>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<positions_copy_stride_three>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<positions_copy_stride_three>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]
      , sml::state<positions_count_publish_decision> <= sml::state<positions_copy_stride_one>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
      , sml::state<errored> <= sml::state<positions_copy_stride_one>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<positions_copy_stride_one>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<positions_copy_stride_one>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<positions_copy_stride_one>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]
      , sml::state<positions_count_publish_decision> <= sml::state<positions_generate_seeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
      , sml::state<errored> <= sml::state<positions_generate_seeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<positions_generate_seeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<positions_generate_seeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<positions_generate_seeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]
      , sml::state<positions_count_publish_decision> <= sml::state<positions_generate_unseeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
      , sml::state<errored> <= sml::state<positions_generate_unseeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<positions_generate_unseeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<positions_generate_unseeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<positions_generate_unseeded>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]

      , sml::state<output_mode_decision> <= sml::state<positions_count_publish_decision>
          + sml::completion<event::batch_runtime> [ guard::positions_count_out_present{} ]
          / action::publish_positions_count
      , sml::state<output_mode_decision> <= sml::state<positions_count_publish_decision>
          + sml::completion<event::batch_runtime> [ guard::positions_count_out_absent{} ]

      //------------------------------------------------------------------------------//
      , sml::state<output_mask_all> <= sml::state<output_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::output_mode_all{} ]
          / action::set_output_mask_all
      , sml::state<output_mask_copy> <= sml::state<output_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::output_mode_copy{} ]
          / action::copy_output_mask
      , sml::state<output_mask_last> <= sml::state<output_mode_decision>
          + sml::completion<event::batch_runtime> [ guard::output_mode_last{} ]
          / action::set_output_mask_last
      , sml::state<errored> <= sml::state<output_mode_decision>
          + sml::completion<event::batch_runtime>
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<output_counting> <= sml::state<output_mask_all>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
          / action::count_outputs_total
      , sml::state<errored> <= sml::state<output_mask_all>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<output_mask_all>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<output_mask_all>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<output_mask_all>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]
      , sml::state<output_counting> <= sml::state<output_mask_copy>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
          / action::count_outputs_total
      , sml::state<errored> <= sml::state<output_mask_copy>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<output_mask_copy>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<output_mask_copy>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<output_mask_copy>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]
      , sml::state<output_counting> <= sml::state<output_mask_last>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
          / action::count_outputs_total
      , sml::state<errored> <= sml::state<output_mask_last>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<output_mask_last>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<output_mask_last>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<output_mask_last>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]

      , sml::state<outputs_total_publish_decision> <= sml::state<output_counting>
          + sml::completion<event::batch_runtime> [ guard::phase_result_ok{} ]
      , sml::state<errored> <= sml::state<output_counting>
          + sml::completion<event::batch_runtime> [ guard::phase_result_invalid_request_error{} ]
      , sml::state<errored> <= sml::state<output_counting>
          + sml::completion<event::batch_runtime> [ guard::phase_result_backend_error{} ]
      , sml::state<errored> <= sml::state<output_counting>
          + sml::completion<event::batch_runtime> [ guard::phase_result_internal_error{} ]
      , sml::state<errored> <= sml::state<output_counting>
          + sml::completion<event::batch_runtime> [ guard::phase_result_unknown_error{} ]

      , sml::state<single_output_decision> <= sml::state<outputs_total_publish_decision>
          + sml::completion<event::batch_runtime> [ guard::outputs_total_out_present{} ]
          / action::publish_outputs_total
      , sml::state<single_output_decision> <= sml::state<outputs_total_publish_decision>
          + sml::completion<event::batch_runtime> [ guard::outputs_total_out_absent{} ]

      //------------------------------------------------------------------------------//
      , sml::state<continuity_decision> <= sml::state<single_output_decision>
          + sml::completion<event::batch_runtime> [ guard::single_output_check_skipped{} ]
      , sml::state<single_output_probe> <= sml::state<single_output_decision>
          + sml::completion<event::batch_runtime> [ guard::single_output_check_required{} ]
          / action::probe_single_output_per_seq
      , sml::state<continuity_decision> <= sml::state<single_output_probe>
          + sml::completion<event::batch_runtime> [ guard::single_output_probe_ok{} ]
      , sml::state<errored> <= sml::state<single_output_probe>
          + sml::completion<event::batch_runtime>
          [ guard::single_output_probe_invalid_request{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<single_output_probe>
          + sml::completion<event::batch_runtime>
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<continuity_decision>
          + sml::completion<event::batch_runtime> [ guard::continuity_check_skipped{} ]
      , sml::state<continuity_probe> <= sml::state<continuity_decision>
          + sml::completion<event::batch_runtime> [ guard::continuity_check_required{} ]
          / action::probe_continuity
      , sml::state<done> <= sml::state<continuity_probe>
          + sml::completion<event::batch_runtime> [ guard::continuity_probe_ok{} ]
      , sml::state<errored> <= sml::state<continuity_probe>
          + sml::completion<event::batch_runtime>
          [ guard::continuity_probe_invalid_request{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<continuity_probe>
          + sml::completion<event::batch_runtime>
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<event::batch_runtime>
          [ guard::done_callback_present{} ]
          / action::publish_done
      , sml::state<ready> <= sml::state<done> + sml::completion<event::batch_runtime>
          [ guard::done_callback_absent{} ]
          / action::publish_done_noop
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::batch_runtime>
          [ guard::error_callback_present{} ]
          / action::publish_error
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::batch_runtime>
          [ guard::error_callback_absent{} ]
          / action::publish_error_noop

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_validation_probe> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_outputs_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_token_counts_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_capacities_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_token_ids_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_seq_payload_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<seq_mode_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<seq_from_masks> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<seq_from_primary_ids> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<seq_default> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<seq_mask_words_publish_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<positions_mode_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<positions_copy_stride_three> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<positions_copy_stride_one> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<positions_seeded_probe> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<positions_unseeded_probe> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<positions_generate_seeded> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<positions_generate_unseeded> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<positions_count_publish_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<output_mode_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<output_mask_all> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<output_mask_copy> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<output_mask_last> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<output_counting> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<outputs_total_publish_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<single_output_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<single_output_probe> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<continuity_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<continuity_probe> + sml::unexpected_event<sml::_>
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

  bool process_event(const event::batch & ev) {
    event::batch_ctx ctx{};
    event::batch_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using Batcher = sm;

}  // namespace emel::token::batcher
