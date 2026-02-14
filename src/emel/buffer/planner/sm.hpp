#pragma once

#include "emel/buffer/planner/actions.hpp"
#include "emel/buffer/planner/events.hpp"
#include "emel/buffer/planner/guards.hpp"
#include "emel/sm.hpp"

namespace emel::buffer::planner {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct idle {};
    struct resetting {};
    struct seeding_leafs {};
    struct counting_references {};
    struct allocating_explicit_inputs {};
    struct planning_nodes {};
    struct releasing_expired {};
    struct finalizing {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::plan> / action::begin_plan = sml::state<resetting>,

      sml::state<resetting> + sml::event<event::reset_done>[guard::no_error{}] /
          action::on_reset_done = sml::state<seeding_leafs>,
      sml::state<resetting> + sml::event<event::reset_done>[guard::has_error{}] / action::no_op =
          sml::state<errored>,
      sml::state<resetting> + sml::event<event::reset_error> / action::record_phase_error =
          sml::state<errored>,

      sml::state<seeding_leafs> + sml::event<event::seed_leafs_done>[guard::no_error{}] /
          action::on_seed_leafs_done = sml::state<counting_references>,
      sml::state<seeding_leafs> + sml::event<event::seed_leafs_done>[guard::has_error{}] /
          action::on_seed_leafs_done = sml::state<errored>,
      sml::state<seeding_leafs> + sml::event<event::seed_leafs_error> / action::record_phase_error =
          sml::state<errored>,

      sml::state<counting_references> + sml::event<event::count_references_done>[guard::no_error{}] /
          action::on_count_references_done = sml::state<allocating_explicit_inputs>,
      sml::state<counting_references> +
          sml::event<event::count_references_done>[guard::has_error{}] /
          action::on_count_references_done = sml::state<errored>,
      sml::state<counting_references> + sml::event<event::count_references_error> /
          action::record_phase_error = sml::state<errored>,

      sml::state<allocating_explicit_inputs> +
          sml::event<event::alloc_explicit_inputs_done>[guard::no_error{}] /
          action::on_alloc_explicit_inputs_done = sml::state<planning_nodes>,
      sml::state<allocating_explicit_inputs> +
          sml::event<event::alloc_explicit_inputs_done>[guard::has_error{}] /
          action::on_alloc_explicit_inputs_done = sml::state<errored>,
      sml::state<allocating_explicit_inputs> + sml::event<event::alloc_explicit_inputs_error> /
          action::record_phase_error = sml::state<errored>,

      sml::state<planning_nodes> + sml::event<event::plan_nodes_done>[guard::no_error{}] /
          action::on_plan_nodes_done = sml::state<releasing_expired>,
      sml::state<planning_nodes> + sml::event<event::plan_nodes_done>[guard::has_error{}] /
          action::on_plan_nodes_done = sml::state<errored>,
      sml::state<planning_nodes> + sml::event<event::plan_nodes_error> /
          action::record_phase_error = sml::state<errored>,

      sml::state<releasing_expired> + sml::event<event::release_expired_done>[guard::no_error{}] /
          action::on_release_expired_done = sml::state<finalizing>,
      sml::state<releasing_expired> + sml::event<event::release_expired_done>[guard::has_error{}] /
          action::on_release_expired_done = sml::state<errored>,
      sml::state<releasing_expired> + sml::event<event::release_expired_error> /
          action::record_phase_error = sml::state<errored>,

      sml::state<finalizing> + sml::event<event::finalize_done>[guard::no_error{}] /
          action::on_finalize_done = sml::state<done>,
      sml::state<finalizing> + sml::event<event::finalize_done>[guard::has_error{}] /
          action::on_finalize_done = sml::state<errored>,
      sml::state<finalizing> + sml::event<event::finalize_error> / action::record_phase_error =
          sml::state<errored>,

      sml::state<resetting> + sml::event<events::plan_error> / action::on_plan_error =
          sml::state<idle>,
      sml::state<seeding_leafs> + sml::event<events::plan_error> / action::on_plan_error =
          sml::state<idle>,
      sml::state<counting_references> + sml::event<events::plan_error> / action::on_plan_error =
          sml::state<idle>,
      sml::state<allocating_explicit_inputs> + sml::event<events::plan_error> /
          action::on_plan_error = sml::state<idle>,
      sml::state<planning_nodes> + sml::event<events::plan_error> / action::on_plan_error =
          sml::state<idle>,
      sml::state<releasing_expired> + sml::event<events::plan_error> / action::on_plan_error =
          sml::state<idle>,
      sml::state<finalizing> + sml::event<events::plan_error> / action::on_plan_error =
          sml::state<idle>,
      sml::state<errored> + sml::event<events::plan_error> / action::on_plan_error =
          sml::state<idle>,

      sml::state<done> + sml::event<events::plan_done> / action::on_plan_done = sml::state<idle>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::plan & ev) {
    if (!base_type::process_event(ev)) {
      return false;
    }

    if (!run_phase<event::reset_done>()) return finalize_error();
    if (!run_phase<event::seed_leafs_done>()) return finalize_error();
    if (!run_phase<event::count_references_done>()) return finalize_error();
    if (!run_phase<event::alloc_explicit_inputs_done>()) return finalize_error();
    if (!run_phase<event::plan_nodes_done>()) return finalize_error();
    if (!run_phase<event::release_expired_done>()) return finalize_error();
    if (!run_phase<event::finalize_done>()) return finalize_error();

    if (context_.pending_error != EMEL_OK) {
      return finalize_error();
    }

    return base_type::process_event(events::plan_done{
      .total_bytes = context_.total_bytes,
    });
  }

  int32_t total_bytes() const noexcept { return context_.total_bytes; }

  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  template <class PhaseDoneEvent>
  bool run_phase() {
    if (!base_type::process_event(PhaseDoneEvent{})) {
      return false;
    }
    return context_.pending_error == EMEL_OK;
  }

  bool finalize_error() {
    const int32_t err = context_.pending_error == EMEL_OK ? EMEL_ERR_BACKEND : context_.pending_error;
    (void)base_type::process_event(events::plan_error{.err = err});
    return false;
  }

  action::context context_{};
};

}  // namespace emel::buffer::planner
