#pragma once

#include <array>
#include <cstdint>

#include "emel/sm.hpp"
#include "emel/text/tokenizer/actions.hpp"
#include "emel/text/tokenizer/detail.hpp"
#include "emel/text/tokenizer/events.hpp"
#include "emel/text/tokenizer/guards.hpp"

namespace emel::text::tokenizer {

struct uninitialized {};
struct binding_preprocessor {};
struct binding_preprocessor_decision {};
struct binding_encoder {};
struct binding_encoder_decision {};
struct idle {};
struct preprocessing {};
struct preprocess_decision {};
struct prefix_decision {};
struct encoding_ready {};
struct encoding_token_fragment {};
struct encoding_raw_fragment {};
struct encoding_raw_decision {};
struct suffix_decision {};
struct finalizing {};
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
        sml::state<binding_preprocessor> <= *sml::state<uninitialized>
                   + sml::event<event::bind_runtime>[ guard::can_bind{} ]
                   / action::begin_bind
      , sml::state<errored> <= sml::state<uninitialized> + sml::event<event::bind_runtime>
                   / action::reject_bind
      , sml::state<errored> <= sml::state<uninitialized> + sml::event<event::tokenize_runtime>
                   / action::reject_invalid

      , sml::state<binding_preprocessor> <= sml::state<idle>
                   + sml::event<event::bind_runtime>[ guard::can_bind{} ]
                   / action::begin_bind
      , sml::state<errored> <= sml::state<idle> + sml::event<event::bind_runtime>
                   / action::reject_bind
      , sml::state<preprocessing> <= sml::state<idle>
                   + sml::event<event::tokenize_runtime>[ guard::can_tokenize{} ]
                   / action::begin_tokenize
      , sml::state<errored> <= sml::state<idle> + sml::event<event::tokenize_runtime>
                   / action::reject_invalid

      , sml::state<binding_preprocessor> <= sml::state<done>
                   + sml::event<event::bind_runtime>[ guard::can_bind{} ]
                   / action::begin_bind
      , sml::state<errored> <= sml::state<done> + sml::event<event::bind_runtime>
                   / action::reject_bind
      , sml::state<preprocessing> <= sml::state<done>
                   + sml::event<event::tokenize_runtime>[ guard::can_tokenize{} ]
                   / action::begin_tokenize
      , sml::state<errored> <= sml::state<done> + sml::event<event::tokenize_runtime>
                   / action::reject_invalid

      , sml::state<binding_preprocessor> <= sml::state<errored>
                   + sml::event<event::bind_runtime>[ guard::can_bind{} ]
                   / action::begin_bind
      , sml::state<errored> <= sml::state<errored> + sml::event<event::bind_runtime>
                   / action::reject_bind
      , sml::state<preprocessing> <= sml::state<errored>
                   + sml::event<event::tokenize_runtime>[ guard::can_tokenize{} ]
                   / action::begin_tokenize
      , sml::state<errored> <= sml::state<errored> + sml::event<event::tokenize_runtime>
                   / action::reject_invalid

      , sml::state<binding_preprocessor> <= sml::state<unexpected>
                   + sml::event<event::bind_runtime>[ guard::can_bind{} ]
                   / action::begin_bind
      , sml::state<unexpected> <= sml::state<unexpected> + sml::event<event::bind_runtime>
                   / action::reject_bind
      , sml::state<preprocessing> <= sml::state<unexpected>
                   + sml::event<event::tokenize_runtime>[ guard::can_tokenize{} ]
                   / action::begin_tokenize
      , sml::state<unexpected> <= sml::state<unexpected> + sml::event<event::tokenize_runtime>
                   / action::reject_invalid

      //------------------------------------------------------------------------------//
      // Bind flow.
      , sml::state<binding_preprocessor_decision> <= sml::state<binding_preprocessor>
                   + sml::completion<event::bind_runtime> / action::bind_preprocessor
      , sml::state<binding_encoder> <= sml::state<binding_preprocessor_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_preprocessor_error_none{} ]
      , sml::state<errored> <= sml::state<binding_preprocessor_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_preprocessor_error_invalid_request{} ]
      , sml::state<errored> <= sml::state<binding_preprocessor_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_preprocessor_error_model_invalid{} ]
      , sml::state<errored> <= sml::state<binding_preprocessor_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_preprocessor_error_backend_error{} ]
      , sml::state<errored> <= sml::state<binding_preprocessor_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_preprocessor_error_unknown{} ]

      , sml::state<binding_encoder_decision> <= sml::state<binding_encoder>
                   + sml::completion<event::bind_runtime> / action::bind_encoder
      , sml::state<idle> <= sml::state<binding_encoder_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_encoder_error_none{} ]
                   / action::mark_bind_success
      , sml::state<errored> <= sml::state<binding_encoder_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_encoder_error_invalid_request{} ]
      , sml::state<errored> <= sml::state<binding_encoder_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_encoder_error_model_invalid{} ]
      , sml::state<errored> <= sml::state<binding_encoder_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_encoder_error_backend_error{} ]
      , sml::state<errored> <= sml::state<binding_encoder_decision>
                   + sml::completion<event::bind_runtime>[ guard::bind_encoder_error_unknown{} ]

      //------------------------------------------------------------------------------//
      // Tokenize flow.
      , sml::state<preprocess_decision> <= sml::state<preprocessing>
                   + sml::completion<event::tokenize_runtime> / action::dispatch_preprocess
      , sml::state<errored> <= sml::state<preprocess_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::preprocess_rejected_no_error{} ]
                   / action::set_backend_error
      , sml::state<errored> <= sml::state<preprocess_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::preprocess_reported_error{} ]
                   / action::set_error_from_preprocess
      , sml::state<errored> <= sml::state<preprocess_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::preprocess_fragment_count_invalid{} ]
                   / action::set_invalid_request_error
      , sml::state<prefix_decision> <= sml::state<preprocess_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::preprocess_success{} ]

      , sml::state<encoding_ready> <= sml::state<prefix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::bos_ready{} ]
                   / action::append_bos
      , sml::state<errored> <= sml::state<prefix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::bos_no_capacity{} ]
                   / action::set_invalid_request_error
      , sml::state<errored> <= sml::state<prefix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::bos_invalid_id{} ]
                   / action::set_invalid_id_error
      , sml::state<encoding_ready> <= sml::state<prefix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::no_prefix{} ]

      , sml::state<suffix_decision> <= sml::state<encoding_ready>
                   + sml::completion<event::tokenize_runtime>[ guard::no_more_fragments{} ]
      , sml::state<errored> <= sml::state<encoding_ready>
                   + sml::completion<event::tokenize_runtime>[ guard::more_fragments_no_capacity{} ]
                   / action::set_invalid_request_error
      , sml::state<errored> <= sml::state<encoding_ready>
                   + sml::completion<event::tokenize_runtime>[ guard::more_fragments_token_invalid{} ]
                   / action::set_invalid_request_error
      , sml::state<encoding_token_fragment> <= sml::state<encoding_ready>
                   + sml::completion<event::tokenize_runtime>[ guard::more_fragments_token_valid{} ]
      , sml::state<encoding_raw_fragment> <= sml::state<encoding_ready>
                   + sml::completion<event::tokenize_runtime>[ guard::more_fragments_raw{} ]

      , sml::state<encoding_ready> <= sml::state<encoding_token_fragment>
                   + sml::completion<event::tokenize_runtime> / action::append_fragment_token

      , sml::state<encoding_raw_decision> <= sml::state<encoding_raw_fragment>
                   + sml::completion<event::tokenize_runtime> / action::dispatch_encode_raw_fragment
      , sml::state<errored> <= sml::state<encoding_raw_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::encode_rejected_no_error{} ]
                   / action::set_invalid_id_error
      , sml::state<errored> <= sml::state<encoding_raw_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::encode_reported_error{} ]
                   / action::set_error_from_encode
      , sml::state<errored> <= sml::state<encoding_raw_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::encode_count_invalid{} ]
                   / action::set_invalid_request_error
      , sml::state<encoding_ready> <= sml::state<encoding_raw_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::encode_success{} ]
                   / action::commit_encoded_fragment

      , sml::state<finalizing> <= sml::state<suffix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::sep_ready{} ]
                   / action::append_sep
      , sml::state<errored> <= sml::state<suffix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::sep_no_capacity{} ]
                   / action::set_invalid_request_error
      , sml::state<errored> <= sml::state<suffix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::sep_invalid_id{} ]
                   / action::set_invalid_id_error
      , sml::state<finalizing> <= sml::state<suffix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::eos_ready{} ]
                   / action::append_eos
      , sml::state<errored> <= sml::state<suffix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::eos_no_capacity{} ]
                   / action::set_invalid_request_error
      , sml::state<errored> <= sml::state<suffix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::eos_invalid_id{} ]
                   / action::set_invalid_id_error
      , sml::state<finalizing> <= sml::state<suffix_decision>
                   + sml::completion<event::tokenize_runtime>[ guard::no_suffix{} ]

      , sml::state<done> <= sml::state<finalizing>
                   + sml::completion<event::tokenize_runtime> / action::finalize

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<unexpected> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_preprocessor> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_preprocessor_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_encoder> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_encoder_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<idle> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<preprocessing> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<preprocess_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<prefix_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encoding_ready> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encoding_token_fragment> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encoding_raw_fragment> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encoding_raw_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<suffix_decision> + sml::unexpected_event<sml::_>
                   / action::on_unexpected
      , sml::state<unexpected> <= sml::state<finalizing> + sml::unexpected_event<sml::_>
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

namespace detail {

inline void dispatch_bind_done(const event::bind &request,
                               const events::tokenizer_bind_done &done_ev,
                               const events::tokenizer_bind_error &) noexcept {
  dispatch_optional_callback(request.owner_sm, request.dispatch_done, done_ev);
}

inline void
dispatch_bind_error(const event::bind &request,
                    const events::tokenizer_bind_done &,
                    const events::tokenizer_bind_error &error_ev) noexcept {
  dispatch_optional_callback(request.owner_sm, request.dispatch_error,
                             error_ev);
}

inline void dispatch_tokenize_done(const event::tokenize &request,
                                   const events::tokenizer_done &done_ev,
                                   const events::tokenizer_error &) noexcept {
  dispatch_optional_callback(request.owner_sm, request.dispatch_done, done_ev);
}

inline void
dispatch_tokenize_error(const event::tokenize &request,
                        const events::tokenizer_done &,
                        const events::tokenizer_error &error_ev) noexcept {
  dispatch_optional_callback(request.owner_sm, request.dispatch_error,
                             error_ev);
}

} // namespace detail

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() = default;

  bool process_event(const event::bind &ev) {
    namespace sml = boost::sml;

    event::bind_ctx runtime_ctx{};
    event::bind_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    const bool ok = this->is(sml::state<idle>);
    const int32_t err = detail::select_error_code(ok, runtime_ctx.err);
    last_error_ = err;

    int32_t error_sink = error_code(error::none);
    detail::write_optional(ev.error_out, error_sink, err);

    const events::tokenizer_bind_done done_ev{&ev};
    const events::tokenizer_bind_error error_ev{&ev, err};
    detail::dispatch_result_callback(ok, ev, done_ev, error_ev,
                                     detail::dispatch_bind_done,
                                     detail::dispatch_bind_error);

    return accepted && ok;
  }

  bool process_event(const event::tokenize &ev) {
    namespace sml = boost::sml;

    event::tokenize_ctx runtime_ctx{};
    runtime_ctx.fragments = tokenize_fragments_.data();
    runtime_ctx.fragment_capacity = tokenize_fragments_.size();
    event::tokenize_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err = detail::select_error_code(ok, runtime_ctx.err);
    last_error_ = err;
    token_count_ = runtime_ctx.token_count;

    int32_t token_count_sink = 0;
    detail::write_optional(ev.token_count_out, token_count_sink,
                           runtime_ctx.token_count);
    int32_t error_sink = error_code(error::none);
    detail::write_optional(ev.error_out, error_sink, err);

    const events::tokenizer_done done_ev{&ev, runtime_ctx.token_count};
    const events::tokenizer_error error_ev{&ev, err};
    detail::dispatch_result_callback(ok, ev, done_ev, error_ev,
                                     detail::dispatch_tokenize_done,
                                     detail::dispatch_tokenize_error);

    return accepted && ok;
  }

  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return last_error_; }
  int32_t token_count() const noexcept { return token_count_; }

private:
  int32_t last_error_ = error_code(error::none);
  int32_t token_count_ = 0;
  std::array<action::fragment, action::k_max_fragments> tokenize_fragments_{};
};

} // namespace emel::text::tokenizer
