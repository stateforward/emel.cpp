#pragma once

#include "emel/buffer/planner/actions.hpp"
#include "emel/buffer/planner/events.hpp"
#include "emel/buffer/planner/guards.hpp"
#include "emel/sm.hpp"

namespace emel::buffer::planner {

using Process = boost::sml::back::process<
  event::reset_done,
  event::reset_error,
  event::seed_leafs_done,
  event::seed_leafs_error,
  event::count_references_done,
  event::count_references_error,
  event::alloc_explicit_inputs_done,
  event::alloc_explicit_inputs_error,
  event::plan_nodes_done,
  event::plan_nodes_error,
  event::release_expired_done,
  event::release_expired_error,
  event::finalize_done,
  event::finalize_error,
  event::split_required_done,
  event::split_required_error,
  events::plan_done,
  events::plan_error>;

template <class Owner>
struct ProcessSupport {
 protected:
  struct ImmediateQueue {
    using container_type = void;
    Owner * owner = nullptr;

    template <class Event>
    void push(const Event & ev) noexcept {
      if (owner == nullptr) {
        return;
      }
      owner->process_event(ev);
    }
  };

  explicit ProcessSupport(Owner * owner) : queue_{owner}, process_{queue_} {}

  ImmediateQueue queue_{};
  Process process_;
};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

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

      sml::state<resetting> + sml::on_entry<event::plan> /
          [](const event::plan & ev, action::context & ctx, process_t & process) noexcept {
            const int32_t err = action::detail::run_reset(ctx);
            if (err != EMEL_OK) {
              process(event::reset_error{
                .err = err,
                .request = &ev,
                .error_out = ev.error_out,
              });
              return;
            }
            process(event::reset_done{
              .request = &ev,
              .error_out = ev.error_out,
            });
          },
      sml::state<resetting> + sml::event<event::reset_done>[guard::has_request{}] =
          sml::state<seeding_leafs>,
      sml::state<resetting> + sml::event<event::reset_done>[guard::missing_request{}] =
          sml::state<errored>,
      sml::state<resetting> + sml::event<event::reset_error> = sml::state<errored>,

      sml::state<seeding_leafs> + sml::on_entry<event::reset_done> /
          [](const event::reset_done & ev, action::context & ctx, process_t & process) noexcept {
            const event::plan * request = ev.request;
            const int32_t err = action::detail::run_seed_leafs(ctx);
            if (err != EMEL_OK) {
              process(event::seed_leafs_error{
                .err = err,
                .request = request,
                .error_out = ev.error_out,
              });
              return;
            }
            process(event::seed_leafs_done{
              .request = request,
              .error_out = ev.error_out,
            });
          },
      sml::state<seeding_leafs> + sml::event<event::seed_leafs_done>[guard::has_request{}] =
          sml::state<counting_references>,
      sml::state<seeding_leafs> + sml::event<event::seed_leafs_done>[guard::missing_request{}] =
          sml::state<errored>,
      sml::state<seeding_leafs> + sml::event<event::seed_leafs_error> = sml::state<errored>,

      sml::state<counting_references> + sml::on_entry<event::seed_leafs_done> /
          [](const event::seed_leafs_done & ev, action::context & ctx, process_t & process) noexcept {
            const event::plan * request = ev.request;
            const int32_t err = action::detail::run_count_references(ctx);
            if (err != EMEL_OK) {
              process(event::count_references_error{
                .err = err,
                .request = request,
                .error_out = ev.error_out,
              });
              return;
            }
            process(event::count_references_done{
              .request = request,
              .error_out = ev.error_out,
            });
          },
      sml::state<counting_references> +
              sml::event<event::count_references_done>[guard::has_request{}] =
          sml::state<allocating_explicit_inputs>,
      sml::state<counting_references> +
              sml::event<event::count_references_done>[guard::missing_request{}] =
          sml::state<errored>,
      sml::state<counting_references> + sml::event<event::count_references_error> =
          sml::state<errored>,

      sml::state<allocating_explicit_inputs> + sml::on_entry<event::count_references_done> /
          [](const event::count_references_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::plan * request = ev.request;
            const int32_t err = action::detail::run_alloc_explicit_inputs(ctx);
            if (err != EMEL_OK) {
              process(event::alloc_explicit_inputs_error{
                .err = err,
                .request = request,
                .error_out = ev.error_out,
              });
              return;
            }
            process(event::alloc_explicit_inputs_done{
              .request = request,
              .error_out = ev.error_out,
            });
          },
      sml::state<allocating_explicit_inputs> +
              sml::event<event::alloc_explicit_inputs_done>[guard::has_request{}] =
          sml::state<planning_nodes>,
      sml::state<allocating_explicit_inputs> +
              sml::event<event::alloc_explicit_inputs_done>[guard::missing_request{}] =
          sml::state<errored>,
      sml::state<allocating_explicit_inputs> + sml::event<event::alloc_explicit_inputs_error> =
          sml::state<errored>,

      sml::state<planning_nodes> + sml::on_entry<event::alloc_explicit_inputs_done> /
          [](const event::alloc_explicit_inputs_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::plan * request = ev.request;
            const int32_t err = action::detail::run_plan_nodes(ctx);
            if (err != EMEL_OK) {
              process(event::plan_nodes_error{
                .err = err,
                .request = request,
                .error_out = ev.error_out,
              });
              return;
            }
            process(event::plan_nodes_done{
              .request = request,
              .error_out = ev.error_out,
            });
          },
      sml::state<planning_nodes> + sml::event<event::plan_nodes_done>[guard::has_request{}] =
          sml::state<releasing_expired>,
      sml::state<planning_nodes> + sml::event<event::plan_nodes_done>[guard::missing_request{}] =
          sml::state<errored>,
      sml::state<planning_nodes> + sml::event<event::plan_nodes_error> =
          sml::state<errored>,

      sml::state<releasing_expired> + sml::on_entry<event::plan_nodes_done> /
          [](const event::plan_nodes_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::plan * request = ev.request;
            const int32_t err = action::detail::run_release_expired(ctx);
            if (err != EMEL_OK) {
              process(event::release_expired_error{
                .err = err,
                .request = request,
                .error_out = ev.error_out,
              });
              return;
            }
            process(event::release_expired_done{
              .request = request,
              .error_out = ev.error_out,
            });
          },
      sml::state<releasing_expired> +
              sml::event<event::release_expired_done>[guard::has_request{}] =
          sml::state<finalizing>,
      sml::state<releasing_expired> +
              sml::event<event::release_expired_done>[guard::missing_request{}] =
          sml::state<errored>,
      sml::state<releasing_expired> + sml::event<event::release_expired_error> =
          sml::state<errored>,

      sml::state<finalizing> + sml::on_entry<event::release_expired_done> /
          [](const event::release_expired_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::plan * request = ev.request;
            const int32_t err = action::detail::run_finalize(ctx, request);
            if (err != EMEL_OK) {
              process(event::finalize_error{
                .err = err,
                .request = request,
                .error_out = ev.error_out,
              });
              return;
            }
            process(event::finalize_done{
              .request = request,
              .error_out = ev.error_out,
            });
          },
      sml::state<finalizing> + sml::event<event::finalize_done>[guard::has_request{}] =
          sml::state<splitting_required>,
      sml::state<finalizing> + sml::event<event::finalize_done>[guard::missing_request{}] =
          sml::state<errored>,
      sml::state<finalizing> + sml::event<event::finalize_error> = sml::state<errored>,

      sml::state<splitting_required> + sml::on_entry<event::finalize_done> /
          [](const event::finalize_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::plan * request = ev.request;
            const int32_t err = action::detail::run_split_required(ctx, request);
            if (err != EMEL_OK) {
              process(event::split_required_error{
                .err = err,
                .request = request,
                .error_out = ev.error_out,
              });
              return;
            }
            process(event::split_required_done{
              .request = request,
              .error_out = ev.error_out,
            });
          },
      sml::state<splitting_required> +
              sml::event<event::split_required_done>[guard::has_request{}] = sml::state<done>,
      sml::state<splitting_required> +
              sml::event<event::split_required_done>[guard::missing_request{}] =
          sml::state<errored>,
      sml::state<splitting_required> + sml::event<event::split_required_error> =
          sml::state<errored>,

      sml::state<done> + sml::on_entry<event::split_required_done> /
          [](const event::split_required_done & ev, action::context & ctx,
             process_t & process) noexcept {
            events::plan_done done_event{
              .total_bytes = ctx.total_bytes,
              .error_out = ev.error_out,
            };
            process(done_event);
            const event::plan * request = ev.request;
            if (request != nullptr && request->dispatch_done != nullptr) {
              (void)request->dispatch_done(request->owner_sm, done_event);
            }
          },
      sml::state<done> + sml::event<events::plan_done> / action::on_plan_done =
          sml::state<idle>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_INVALID_ARGUMENT;
            int32_t * error_out = nullptr;
            const event::plan * request = nullptr;
            void * owner_sm = nullptr;
            bool (*dispatch_error)(void *, const events::plan_error &) = nullptr;

            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.error_out; }) {
              error_out = ev.error_out;
            }
            if constexpr (requires { ev.request; }) {
              request = ev.request;
            }
            if constexpr (requires { ev.owner_sm; }) {
              owner_sm = ev.owner_sm;
            }
            if constexpr (requires { ev.dispatch_error; }) {
              dispatch_error = ev.dispatch_error;
            }
            if (request != nullptr) {
              owner_sm = request->owner_sm;
              dispatch_error = request->dispatch_error;
            }

            events::plan_error error_event{
              .err = err,
              .error_out = error_out,
            };
            process(error_event);
            if (dispatch_error != nullptr) {
              (void)dispatch_error(owner_sm, error_event);
            }
          },
      sml::state<errored> + sml::event<events::plan_error> / action::on_plan_error =
          sml::state<idle>,

      sml::state<idle> + sml::event<sml::_> / action::on_unexpected = sml::state<errored>,
      sml::state<resetting> + sml::event<sml::_> / action::on_unexpected = sml::state<errored>,
      sml::state<seeding_leafs> + sml::event<sml::_> / action::on_unexpected =
          sml::state<errored>,
      sml::state<counting_references> + sml::event<sml::_> / action::on_unexpected =
          sml::state<errored>,
      sml::state<allocating_explicit_inputs> + sml::event<sml::_> / action::on_unexpected =
          sml::state<errored>,
      sml::state<planning_nodes> + sml::event<sml::_> / action::on_unexpected =
          sml::state<errored>,
      sml::state<releasing_expired> + sml::event<sml::_> / action::on_unexpected =
          sml::state<errored>,
      sml::state<finalizing> + sml::event<sml::_> / action::on_unexpected =
          sml::state<errored>,
      sml::state<splitting_required> + sml::event<sml::_> / action::on_unexpected =
          sml::state<errored>,
      sml::state<done> + sml::event<sml::_> / action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::event<sml::_> / action::on_unexpected =
          sml::state<errored>
    );
  }
};

struct sm : private ProcessSupport<sm>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : ProcessSupport<sm>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  int32_t total_bytes() const noexcept { return context_.total_bytes; }

 private:
 action::context context_{};
};

}  // namespace emel::buffer::planner
