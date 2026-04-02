#pragma once
// benchmark: designed

#include "emel/generator/prefill/actions.hpp"
#include "emel/generator/prefill/guards.hpp"
#include "emel/sm.hpp"

namespace emel::generator::prefill {

struct idle {};
struct slots {};
struct slots_decision {};
struct snapshot {};
struct snapshot_decision {};
struct contract_runtime_decision {};
struct contract_flash_decision {};
struct contract_nonflash_decision {};
struct compute_result_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<slots> <= *sml::state<idle> + sml::event<event::run>

      , sml::state<slots_decision> <= sml::state<slots> + sml::completion<event::run>
                 / action::request_slots

      , sml::state<snapshot> <= sml::state<slots_decision> + sml::completion<event::run>
                 [ guard::slots_ok{} ]

      , sml::state<idle> <= sml::state<slots_decision> + sml::completion<event::run>
                 [ guard::slots_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<idle> <= sml::state<slots_decision> + sml::completion<event::run>
                 [ guard::slots_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<snapshot_decision> <= sml::state<snapshot> + sml::completion<event::run>
                 / action::request_memory_snapshot

      , sml::state<contract_runtime_decision> <= sml::state<snapshot_decision>
                 + sml::completion<event::run>
                 [ guard::snapshot_ok{} ]

      , sml::state<idle> <= sml::state<snapshot_decision> + sml::completion<event::run>
                 [ guard::snapshot_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<idle> <= sml::state<snapshot_decision> + sml::completion<event::run>
                 [ guard::snapshot_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<contract_flash_decision> <= sml::state<contract_runtime_decision>
                 + sml::completion<event::run>
                 [ guard::flash_runtime_supported{} ]

      , sml::state<contract_nonflash_decision> <= sml::state<contract_runtime_decision>
                 + sml::completion<event::run>
                 [ guard::nonflash_runtime_required{} ]

      , sml::state<compute_result_decision> <= sml::state<contract_flash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits_with_chunk8_q8_k{} ]
                 / action::request_contract_flash_materialized_chunk8_q8_k

      , sml::state<compute_result_decision> <= sml::state<contract_flash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits_with_chunk4_packed_q8_0{} ]
                 / action::request_contract_flash_materialized_chunk4_packed_q8_0

      , sml::state<compute_result_decision> <= sml::state<contract_flash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits_with_chunk4_q8_k{} ]
                 / action::request_contract_flash_materialized_chunk4_q8_k

      , sml::state<compute_result_decision> <= sml::state<contract_flash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits_with_scalar{} ]
                 / action::request_contract_flash_materialized_scalar

      , sml::state<compute_result_decision> <= sml::state<contract_flash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax_with_chunk8_q8_k{} ]
                 / action::request_contract_flash_preselected_chunk8_q8_k

      , sml::state<compute_result_decision> <= sml::state<contract_flash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax_with_chunk4_packed_q8_0{} ]
                 / action::request_contract_flash_preselected_chunk4_packed_q8_0

      , sml::state<compute_result_decision> <= sml::state<contract_flash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax_with_chunk4_q8_k{} ]
                 / action::request_contract_flash_preselected_chunk4_q8_k

      , sml::state<compute_result_decision> <= sml::state<contract_flash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax_with_scalar{} ]
                 / action::request_contract_flash_preselected_scalar

      , sml::state<compute_result_decision> <= sml::state<contract_nonflash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits_with_chunk8_q8_k{} ]
                 / action::request_contract_nonflash_materialized_chunk8_q8_k

      , sml::state<compute_result_decision> <= sml::state<contract_nonflash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits_with_chunk4_packed_q8_0{} ]
                 / action::request_contract_nonflash_materialized_chunk4_packed_q8_0

      , sml::state<compute_result_decision> <= sml::state<contract_nonflash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits_with_chunk4_q8_k{} ]
                 / action::request_contract_nonflash_materialized_chunk4_q8_k

      , sml::state<compute_result_decision> <= sml::state<contract_nonflash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits_with_scalar{} ]
                 / action::request_contract_nonflash_materialized_scalar

      , sml::state<compute_result_decision> <= sml::state<contract_nonflash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax_with_chunk8_q8_k{} ]
                 / action::request_contract_nonflash_preselected_chunk8_q8_k

      , sml::state<compute_result_decision> <= sml::state<contract_nonflash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax_with_chunk4_packed_q8_0{} ]
                 / action::request_contract_nonflash_preselected_chunk4_packed_q8_0

      , sml::state<compute_result_decision> <= sml::state<contract_nonflash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax_with_chunk4_q8_k{} ]
                 / action::request_contract_nonflash_preselected_chunk4_q8_k

      , sml::state<compute_result_decision> <= sml::state<contract_nonflash_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax_with_scalar{} ]
                 / action::request_contract_nonflash_preselected_scalar

      , sml::state<idle> <= sml::state<compute_result_decision> + sml::completion<event::run>
                 [ guard::compute_ok{} ]
                 / action::mark_prefill_cached

      , sml::state<idle> <= sml::state<compute_result_decision> + sml::completion<event::run>
                 [ guard::compute_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<idle> <= sml::state<compute_result_decision> + sml::completion<event::run>
                 [ guard::compute_backend_error{} ]
                 / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<idle> <= sml::state<idle> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<slots> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<slots_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<snapshot> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<snapshot_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<contract_runtime_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<contract_flash_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<contract_nonflash_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<compute_result_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  explicit sm(const action::context & context_in) : base_type(context_in) {}

  bool process_event(const event::run & ev) { return base_type::process_event(ev); }
};

}  // namespace emel::generator::prefill
