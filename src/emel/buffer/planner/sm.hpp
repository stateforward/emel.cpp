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
    struct splitting_required {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::plan>[guard::valid_plan{}] / action::begin_plan =
          sml::state<resetting>,
      sml::state<idle> + sml::event<event::plan> / action::reject_plan = sml::state<errored>,

      sml::state<resetting> / action::run_reset = sml::state<seeding_leafs>,
      sml::state<seeding_leafs> [guard::phase_failed] = sml::state<errored>,
      sml::state<seeding_leafs> [guard::phase_ok] / action::run_seed_leafs =
          sml::state<counting_references>,
      sml::state<counting_references> [guard::phase_failed] = sml::state<errored>,
      sml::state<counting_references> [guard::phase_ok] / action::run_count_references =
          sml::state<allocating_explicit_inputs>,
      sml::state<allocating_explicit_inputs> [guard::phase_failed] = sml::state<errored>,
      sml::state<allocating_explicit_inputs> [guard::phase_ok] /
          action::run_alloc_explicit_inputs = sml::state<planning_nodes>,
      sml::state<planning_nodes> [guard::phase_failed] = sml::state<errored>,
      sml::state<planning_nodes> [guard::phase_ok] / action::run_plan_nodes =
          sml::state<releasing_expired>,
      sml::state<releasing_expired> [guard::phase_failed] = sml::state<errored>,
      sml::state<releasing_expired> [guard::phase_ok] / action::run_release_expired =
          sml::state<finalizing>,
      sml::state<finalizing> [guard::phase_failed] = sml::state<errored>,
      sml::state<finalizing> [guard::phase_ok] / action::run_finalize =
          sml::state<splitting_required>,
      sml::state<splitting_required> [guard::phase_failed] = sml::state<errored>,
      sml::state<splitting_required> [guard::phase_ok] / action::run_split_required =
          sml::state<done>,
      sml::state<done> [guard::phase_failed] = sml::state<errored>,
      sml::state<done> [guard::phase_ok] / action::on_plan_done =
          sml::state<idle>,

      sml::state<errored> [guard::always] / action::on_plan_error = sml::state<idle>,

      sml::state<resetting> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<seeding_leafs> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<counting_references> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<allocating_explicit_inputs> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<planning_nodes> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<releasing_expired> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<finalizing> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<splitting_required> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<done> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>,
      sml::state<errored> + sml::event<event::plan> / action::on_unexpected =
          sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::plan & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    if (context_.phase_error == EMEL_OK) {
      action::detail::copy_plan_outputs(context_, ev);
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_OK;
      }
      if (ev.dispatch_done != nullptr) {
        events::plan_done done{
          .total_bytes = context_.total_bytes,
          .error_out = ev.error_out,
        };
        (void)ev.dispatch_done(ev.owner_sm, done);
      }
    } else {
      const int32_t err = action::detail::normalize_error(context_.phase_error);
      if (ev.error_out != nullptr) {
        *ev.error_out = err;
      }
      if (ev.dispatch_error != nullptr) {
        events::plan_error error{
          .err = err,
          .error_out = ev.error_out,
        };
        (void)ev.dispatch_error(ev.owner_sm, error);
      }
    }
    return emel::detail::normalize_event_result(ev, accepted);
  }

 int32_t total_bytes() const noexcept { return context_.total_bytes; }

 private:
  action::context context_{};
};

}  // namespace emel::buffer::planner
