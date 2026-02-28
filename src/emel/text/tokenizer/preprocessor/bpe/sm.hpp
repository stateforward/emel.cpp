#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/sm.hpp"
#include "emel/text/tokenizer/preprocessor/bpe/actions.hpp"
#include "emel/text/tokenizer/preprocessor/bpe/guards.hpp"
#include "emel/text/tokenizer/preprocessor/detail.hpp"

namespace emel::text::tokenizer::preprocessor::bpe {

namespace pdetail = emel::text::tokenizer::preprocessor::detail;

struct idle {};
struct preparing {};
struct build_specials_decision {};
struct partitioning_select {};
struct partitioning_bpe_no_specials {};
struct partitioning_bpe_with_specials {};
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
        sml::state<preparing> <= *sml::state<idle>
                   + sml::event<event::preprocess_runtime>[ guard::valid_request{} ]
                   / action::begin_preprocess
      , sml::state<errored> <= sml::state<idle>
                   + sml::event<event::preprocess_runtime>[ guard::invalid_request{} ]
                   / action::reject_invalid

      , sml::state<preparing> <= sml::state<done>
                   + sml::event<event::preprocess_runtime>[ guard::valid_request{} ]
                   / action::begin_preprocess
      , sml::state<errored> <= sml::state<done>
                   + sml::event<event::preprocess_runtime>[ guard::invalid_request{} ]
                   / action::reject_invalid

      , sml::state<preparing> <= sml::state<errored>
                   + sml::event<event::preprocess_runtime>[ guard::valid_request{} ]
                   / action::begin_preprocess
      , sml::state<errored> <= sml::state<errored>
                   + sml::event<event::preprocess_runtime>[ guard::invalid_request{} ]
                   / action::reject_invalid

      , sml::state<preparing> <= sml::state<unexpected>
                   + sml::event<event::preprocess_runtime>[ guard::valid_request{} ]
                   / action::begin_preprocess
      , sml::state<errored> <= sml::state<unexpected>
                   + sml::event<event::preprocess_runtime>[ guard::invalid_request{} ]
                   / action::reject_invalid

      //------------------------------------------------------------------------------//
      // Internal phase flow.
      , sml::state<build_specials_decision> <= sml::state<preparing>
                   + sml::completion<event::preprocess_runtime>
                   / action::build_specials

      , sml::state<errored> <= sml::state<build_specials_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::phase_failed{} ]
                   / action::ensure_last_error
      , sml::state<partitioning_select> <= sml::state<build_specials_decision>
                   + sml::completion<event::preprocess_runtime>[ guard::phase_ok{} ]

      , sml::state<partitioning_bpe_no_specials> <= sml::state<partitioning_select>
                   + sml::completion<event::preprocess_runtime>[ guard::no_specials{} ]
      , sml::state<partitioning_bpe_with_specials> <= sml::state<partitioning_select>
                   + sml::completion<event::preprocess_runtime>[ guard::has_specials{} ]

      , sml::state<partition_decision> <= sml::state<partitioning_bpe_no_specials>
                   + sml::completion<event::preprocess_runtime>
                   / action::partition_bpe_no_specials
      , sml::state<partition_decision> <= sml::state<partitioning_bpe_with_specials>
                   + sml::completion<event::preprocess_runtime>
                   / action::partition_bpe_with_specials

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
      , sml::state<unexpected> <= sml::state<preparing> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<build_specials_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<partitioning_select> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<partitioning_bpe_no_specials> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<partitioning_bpe_with_specials> + sml::unexpected_event<sml::_>
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

}  // namespace emel::text::tokenizer::preprocessor::bpe
