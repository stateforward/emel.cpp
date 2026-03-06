#pragma once

#include "emel/token/batcher/actions.hpp"
#include "emel/error/error.hpp"
#include "emel/token/batcher/context.hpp"
#include "emel/token/batcher/errors.hpp"
#include "emel/token/batcher/events.hpp"

namespace emel::token::batcher::guard {

inline bool phase_error_is(const event::batch_runtime & ev,
                           const emel::error::type code_value) noexcept {
  return ev.ctx.err == code_value;
}

struct phase_result_ok {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return phase_error_is(ev, emel::error::cast(error::none));
  }
};

struct phase_result_invalid_request_error {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return phase_error_is(ev, emel::error::cast(error::invalid_request));
  }
};

struct phase_result_backend_error {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return phase_error_is(ev, emel::error::cast(error::backend_error));
  }
};

struct phase_result_internal_error {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return phase_error_is(ev, emel::error::cast(error::internal_error));
  }
};

struct phase_result_unknown_error {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    const emel::error::type err = ev.ctx.err;
    return err != emel::error::cast(error::none) &&
           err != emel::error::cast(error::invalid_request) &&
           err != emel::error::cast(error::backend_error) &&
           err != emel::error::cast(error::internal_error);
  }
};

struct request_outputs_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::required_outputs_present(ev.request);
  }
};

struct request_outputs_missing {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !request_outputs_present{}(ev);
  }
};

struct request_token_counts_valid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::token_counts_valid(ev.request);
  }
};

struct request_token_counts_invalid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !request_token_counts_valid{}(ev);
  }
};

struct request_capacities_valid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::capacities_valid(ev.request);
  }
};

struct request_capacities_invalid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !request_capacities_valid{}(ev);
  }
};

struct request_token_ids_in_vocab {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::token_ids_in_vocab(ev.request);
  }
};

struct request_token_ids_out_of_vocab {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !request_token_ids_in_vocab{}(ev);
  }
};

struct request_seq_payload_valid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::seq_payload_valid(ev.request);
  }
};

struct request_seq_payload_invalid {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !request_seq_payload_valid{}(ev);
  }
};

struct positions_seeded_probe_ok {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct positions_seeded_probe_backend_error {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::backend_error);
  }
};

struct positions_seeded_probe_invalid_request {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::invalid_request);
  }
};

struct positions_unseeded_probe_ok {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct positions_unseeded_probe_invalid_request {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::invalid_request);
  }
};

struct single_output_probe_ok {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct single_output_probe_invalid_request {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::invalid_request);
  }
};

struct continuity_probe_ok {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::none);
  }
};

struct continuity_probe_invalid_request {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.ctx.err == emel::error::cast(error::invalid_request);
  }
};

struct seq_mode_masks {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::has_seq_masks_input(ev.request);
  }
};

struct seq_mode_primary_ids {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !emel::token::batcher::detail::has_seq_masks_input(ev.request) &&
           emel::token::batcher::detail::has_seq_primary_input(ev.request);
  }
};

struct seq_mode_default {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !emel::token::batcher::detail::has_seq_masks_input(ev.request) &&
           !emel::token::batcher::detail::has_seq_primary_input(ev.request);
  }
};

struct positions_mode_stride_three {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) == 3;
  }
};

struct positions_mode_stride_one {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) == 1;
  }
};

struct positions_mode_generate_seeded {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) == 0 &&
           ev.request.resolve_position_seed != nullptr;
  }
};

struct positions_mode_generate_unseeded {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) == 0 &&
           ev.request.resolve_position_seed == nullptr;
  }
};

struct output_mode_all {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.output_all;
  }
};

struct output_mode_copy {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !ev.request.output_all &&
           emel::token::batcher::detail::has_output_mask_input(ev.request);
  }
};

struct output_mode_last {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !ev.request.output_all &&
           !emel::token::batcher::detail::has_output_mask_input(ev.request);
  }
};

struct single_output_check_required {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.enforce_single_output_per_seq;
  }
};

struct single_output_check_skipped {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !ev.request.enforce_single_output_per_seq;
  }
};

struct continuity_check_required {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) <= 1;
  }
};

struct continuity_check_skipped {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return emel::token::batcher::detail::positions_stride(ev.request) > 1;
  }
};

struct seq_mask_words_out_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.seq_mask_words_out != nullptr;
  }
};

struct seq_mask_words_out_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !seq_mask_words_out_present{}(ev);
  }
};

struct positions_count_out_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.positions_count_out != nullptr;
  }
};

struct positions_count_out_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !positions_count_out_present{}(ev);
  }
};

struct outputs_total_out_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return ev.request.outputs_total_out != nullptr;
  }
};

struct outputs_total_out_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !outputs_total_out_present{}(ev);
  }
};

struct done_callback_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct done_callback_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !done_callback_present{}(ev);
  }
};

struct error_callback_present {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const event::batch_runtime & ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

}  // namespace emel::token::batcher::guard
