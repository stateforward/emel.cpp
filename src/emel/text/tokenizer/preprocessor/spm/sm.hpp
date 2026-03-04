#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/sm.hpp"
#include "emel/text/tokenizer/preprocessor/spm/actions.hpp"
#include "emel/text/tokenizer/preprocessor/spm/guards.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"

namespace emel::text::tokenizer::preprocessor::spm {

namespace pdetail = emel::text::tokenizer::preprocessor::detail;

struct idle {};
struct request_buffer_decision {};
struct request_capacity_nonzero_decision {};
struct request_capacity_limit_decision {};
struct preparing {};
struct build_specials_decision {};
struct partition_parse_special_decision {};
struct partitioning_non_bpe_parse_special {};
struct partitioning_non_bpe_skip_special {};
struct partition_decision {};
struct done {};
struct errored {};
struct unexpected {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // External request validation.
        sml::state<request_buffer_decision> <= *sml::state<idle>
                   + sml::event<event::preprocess_runtime>
      , sml::state<request_buffer_decision> <= sml::state<done>
                   + sml::event<event::preprocess_runtime>
      , sml::state<request_buffer_decision> <= sml::state<errored>
                   + sml::event<event::preprocess_runtime>
      , sml::state<request_buffer_decision> <= sml::state<unexpected>
                   + sml::event<event::preprocess_runtime>

      , sml::state<request_capacity_nonzero_decision> <= sml::state<request_buffer_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::fragments_buffer_present{} ]
      , sml::state<errored> <= sml::state<request_buffer_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::fragments_buffer_missing{} ]
                   / action::reject_invalid
      , sml::state<errored> <= sml::state<request_buffer_decision>
                   + sml::completion<event::preprocess_runtime>
                   / action::reject_invalid

      , sml::state<request_capacity_limit_decision> <= sml::state<request_capacity_nonzero_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::fragments_capacity_nonzero{} ]
      , sml::state<errored> <= sml::state<request_capacity_nonzero_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::fragments_capacity_zero{} ]
                   / action::reject_invalid
      , sml::state<errored> <= sml::state<request_capacity_nonzero_decision>
                   + sml::completion<event::preprocess_runtime>
                   / action::reject_invalid

      , sml::state<preparing> <= sml::state<request_capacity_limit_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::fragments_capacity_within_limit{} ]
                   / action::begin_preprocess
      , sml::state<errored> <= sml::state<request_capacity_limit_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::fragments_capacity_exceeds_limit{} ]
                   / action::reject_invalid
      , sml::state<errored> <= sml::state<request_capacity_limit_decision>
                   + sml::completion<event::preprocess_runtime>
                   / action::reject_invalid

      //------------------------------------------------------------------------------//
      // Internal phase flow.
      , sml::state<build_specials_decision> <= sml::state<preparing>
                   + sml::completion<event::preprocess_runtime>
                   / action::build_specials

      , sml::state<errored> <= sml::state<build_specials_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::phase_failed{} ]
                   / action::ensure_last_error
      , sml::state<partition_parse_special_decision> <= sml::state<build_specials_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::phase_ok{} ]

      , sml::state<partitioning_non_bpe_parse_special> <= sml::state<partition_parse_special_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::parse_special_enabled{} ]
      , sml::state<partitioning_non_bpe_skip_special> <= sml::state<partition_parse_special_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::parse_special_disabled{} ]
      , sml::state<errored> <= sml::state<partition_parse_special_decision>
                   + sml::completion<event::preprocess_runtime>
                   / action::ensure_last_error

      , sml::state<partition_decision> <= sml::state<partitioning_non_bpe_parse_special>
                   + sml::completion<event::preprocess_runtime>
                   / action::partition_non_bpe_parse_special
      , sml::state<partition_decision> <= sml::state<partitioning_non_bpe_skip_special>
                   + sml::completion<event::preprocess_runtime>
                   / action::partition_non_bpe_skip_special

      , sml::state<errored> <= sml::state<partition_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::phase_failed{} ]
                   / action::ensure_last_error
      , sml::state<done> <= sml::state<partition_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::phase_ok{} ]
                   / action::mark_done

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<unexpected> <= sml::state<idle> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<request_buffer_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<request_capacity_nonzero_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<request_capacity_limit_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<preparing> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<build_specials_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<partition_parse_special_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<partitioning_non_bpe_parse_special> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<partitioning_non_bpe_skip_special> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<partition_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<done> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<errored> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unexpected> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() = default;

  bool process_event(const event::preprocess & ev) {
    namespace sml = boost::sml;

    event::preprocess_ctx runtime_ctx{};
    event::preprocess_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    const bool ok = this->is(sml::state<done>);
    const preprocessor::error err = pdetail::select_error(ok, runtime_ctx.err);
    const int32_t err_code = preprocessor::error_code(err);

    last_error_ = err_code;
    fragment_count_ = runtime_ctx.fragment_count;

    ev.fragment_count_out = runtime_ctx.fragment_count;
    bool preprocessed_sink = false;
    pdetail::write_optional(ev.preprocessed_out, preprocessed_sink,
                            runtime_ctx.preprocessed);
    ev.error_out = err_code;

    const events::preprocess_done done_ev{&ev, runtime_ctx.fragment_count};
    const events::preprocess_error error_ev{&ev, err};
    pdetail::dispatch_result_callback(ok, ev, done_ev, error_ev,
                                      pdetail::dispatch_preprocess_done,
                                      pdetail::dispatch_preprocess_error);

    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return last_error_; }
  size_t fragment_count() const noexcept { return fragment_count_; }

 private:
  int32_t last_error_ = preprocessor::error_code(preprocessor::error::none);
  size_t fragment_count_ = 0;
};

}  // namespace emel::text::tokenizer::preprocessor::spm
