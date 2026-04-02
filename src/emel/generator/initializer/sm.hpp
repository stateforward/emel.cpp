#pragma once

#include "emel/generator/initializer/actions.hpp"
#include "emel/generator/initializer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::generator::initializer {

struct idle {};
struct preparing_backend {};
struct preparing_backend_decision {};
struct binding_conditioner {};
struct binding_conditioner_decision {};
struct initializing_renderer {};
struct initializing_renderer_decision {};
struct reserving_memory {};
struct reserving_memory_decision {};
struct reserving_graph {};
struct reserving_graph_decision {};
struct configuring_sampling_mode_decision {};
struct configuring_sampler {};
struct configuring_sampler_decision {};
struct configure_preselected_argmax {};
struct configure_preselected_argmax_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<preparing_backend> <= *sml::state<idle> + sml::event<event::run>
                 / action::begin_initialize

      , sml::state<binding_conditioner> <= sml::state<preparing_backend_decision>
                 + sml::completion<event::run>
                 [ guard::backend_already_ready{} ]

      , sml::state<preparing_backend_decision> <= sml::state<preparing_backend>
                 + sml::completion<event::run>
                 [ guard::backend_prepare_needed{} ]
                 / action::request_backend_prepare

      , sml::state<binding_conditioner> <= sml::state<preparing_backend_decision>
                 + sml::completion<event::run>
                 [ guard::backend_prepare_ok{} ]
                 / action::accept_prepared_backend

      , sml::state<idle> <= sml::state<preparing_backend_decision>
                 + sml::completion<event::run>
                 [ guard::backend_prepare_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<idle> <= sml::state<preparing_backend_decision>
                 + sml::completion<event::run>
                 [ guard::backend_prepare_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<binding_conditioner_decision> <= sml::state<binding_conditioner>
                 + sml::completion<event::run>
                 / action::request_conditioner_bind

      , sml::state<initializing_renderer> <= sml::state<binding_conditioner_decision>
                 + sml::completion<event::run>
                 [ guard::conditioner_bind_ok{} ]

      , sml::state<idle> <= sml::state<binding_conditioner_decision>
                 + sml::completion<event::run>
                 [ guard::conditioner_bind_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<idle> <= sml::state<binding_conditioner_decision>
                 + sml::completion<event::run>
                 [ guard::conditioner_bind_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<initializing_renderer_decision> <= sml::state<initializing_renderer>
                 + sml::completion<event::run>
                 / action::request_renderer_initialize

      , sml::state<reserving_memory> <= sml::state<initializing_renderer_decision>
                 + sml::completion<event::run>
                 [ guard::renderer_initialize_ok{} ]

      , sml::state<idle> <= sml::state<initializing_renderer_decision>
                 + sml::completion<event::run>
                 [ guard::renderer_initialize_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<idle> <= sml::state<initializing_renderer_decision>
                 + sml::completion<event::run>
                 [ guard::renderer_initialize_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<reserving_memory_decision> <= sml::state<reserving_memory>
                 + sml::completion<event::run>
                 / action::request_memory_reserve

      , sml::state<configuring_sampling_mode_decision> <= sml::state<reserving_memory_decision>
                 + sml::completion<event::run>
                 [ guard::memory_reserve_with_existing_graph{} ]

      , sml::state<reserving_graph> <= sml::state<reserving_memory_decision>
                 + sml::completion<event::run>
                 [ guard::memory_reserve_with_missing_graph{} ]

      , sml::state<idle> <= sml::state<reserving_memory_decision>
                 + sml::completion<event::run>
                 [ guard::memory_reserve_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<idle> <= sml::state<reserving_memory_decision>
                 + sml::completion<event::run>
                 [ guard::memory_reserve_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<reserving_graph_decision> <= sml::state<reserving_graph>
                 + sml::completion<event::run>
                 / action::request_graph_reserve

      , sml::state<configuring_sampling_mode_decision> <= sml::state<reserving_graph_decision>
                 + sml::completion<event::run>
                 [ guard::graph_reserve_ok{} ]

      , sml::state<idle> <= sml::state<reserving_graph_decision>
                 + sml::completion<event::run>
                 [ guard::graph_reserve_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<idle> <= sml::state<reserving_graph_decision>
                 + sml::completion<event::run>
                 [ guard::graph_reserve_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<configuring_sampler> <= sml::state<configuring_sampling_mode_decision>
                 + sml::completion<event::run>
                 [ guard::uses_materialized_logits{} ]

      , sml::state<configure_preselected_argmax> <= sml::state<configuring_sampling_mode_decision>
                 + sml::completion<event::run>
                 [ guard::uses_preselected_argmax{} ]

      , sml::state<configuring_sampler_decision> <= sml::state<configuring_sampler>
                 + sml::completion<event::run>
                 / action::configure_sampler

      , sml::state<idle> <= sml::state<configuring_sampler_decision>
                 + sml::completion<event::run>
                 [ guard::sampler_configured{} ]

      , sml::state<idle> <= sml::state<configuring_sampler_decision>
                 + sml::completion<event::run>
                 [ guard::sampler_config_failed{} ]
                 / action::mark_backend_error

      , sml::state<configure_preselected_argmax_decision> <= sml::state<configure_preselected_argmax>
                 + sml::completion<event::run>
                 / action::configure_preselected_argmax

      , sml::state<idle> <= sml::state<configure_preselected_argmax_decision>
                 + sml::completion<event::run>
                 [ guard::sampler_configured{} ]

      , sml::state<idle> <= sml::state<configure_preselected_argmax_decision>
                 + sml::completion<event::run>
                 [ guard::sampler_config_failed{} ]
                 / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<idle> <= sml::state<idle> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<preparing_backend> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<preparing_backend_decision>
                 + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<binding_conditioner> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<binding_conditioner_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<initializing_renderer> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<initializing_renderer_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<reserving_memory> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<reserving_memory_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<reserving_graph> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<reserving_graph_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<configuring_sampling_mode_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<configuring_sampler> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<configuring_sampler_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<configure_preselected_argmax> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<idle> <= sml::state<configure_preselected_argmax_decision>
                 + sml::unexpected_event<sml::_>
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

}  // namespace emel::generator::initializer
