#include <doctest/doctest.h>
#include <stateforward/sml/utility/dispatch_table.hpp>

#include "emel/sm.hpp"

namespace {

struct dummy_event {
  int32_t * error_out = nullptr;
};

struct owner_probe {
  int32_t calls = 0;

  bool process_event(const dummy_event &) noexcept {
    calls += 1;
    return true;
  }
};

struct runtime_event {
  int id = 0;
  int32_t & marker;
};

struct event_one {
  static constexpr auto id = 1;

  int32_t & marker;

  explicit event_one(const runtime_event & ev) noexcept
    : marker(ev.marker) {}
};

struct event_two {
  static constexpr auto id = 2;

  int32_t & marker;

  explicit event_two(const runtime_event & ev) noexcept
    : marker(ev.marker) {}
};

struct state_idle {};
struct state_seen_one {};

struct effect_mark_one {
  void operator()(const event_one & ev) const noexcept {
    ev.marker = 1;
  }
};

struct effect_mark_two {
  void operator()(const event_two & ev) const noexcept {
    ev.marker = 2;
  }
};

struct dispatch_surface_model {
  auto operator()() const noexcept {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_seen_one> <= *sml::state<state_idle>
          + sml::event<event_one> / effect_mark_one{}
      , sml::X <= sml::state<state_seen_one>
          + sml::event<event_two> / effect_mark_two{}
    );
    // clang-format on
  }
};

struct logger_event {
  int32_t & marker;
};

struct logger_counters {
  int32_t processed = 0;
  int32_t guards = 0;
  int32_t actions = 0;
  int32_t state_changes = 0;

  template <class SM, class TEvent>
  void log_process_event(const TEvent &) noexcept {
    ++processed;
  }

  template <class SM, class TGuard, class TEvent>
  void log_guard(const TGuard &, const TEvent &, bool) noexcept {
    ++guards;
  }

  template <class SM, class TAction, class TEvent>
  void log_action(const TAction &, const TEvent &) noexcept {
    ++actions;
  }

  template <class SM, class TSrcState, class TDstState>
  void log_state_change(const TSrcState &, const TDstState &) noexcept {
    ++state_changes;
  }
};

struct guard_accept {
  bool operator()() const noexcept {
    return true;
  }
};

struct effect_mark_logged {
  void operator()(const logger_event & ev) const noexcept {
    ev.marker = 7;
  }
};

struct state_logger_idle {};
struct state_logger_done {};

struct logger_surface_model {
  auto operator()() const noexcept {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_logger_done> <= *sml::state<state_logger_idle>
          + sml::event<logger_event> [ guard_accept{} ] / effect_mark_logged{}
      , sml::state<state_logger_idle> <= sml::state<state_logger_idle>
          + sml::unexpected_event<sml::_>
    );
    // clang-format on
  }
};

}  // namespace

TEST_CASE("sm_normalize_event_result_handles_error_out") {
  int32_t err = 0;
  dummy_event ok{.error_out = &err};
  CHECK_FALSE(emel::detail::normalize_event_result(ok, false));
  CHECK(emel::detail::normalize_event_result(ok, true));

  err = (1 << 1);
  CHECK_FALSE(emel::detail::normalize_event_result(ok, true));

  struct no_error_event {};
  CHECK(emel::detail::normalize_event_result(no_error_event{}, true));
}

TEST_CASE("sm_process_support_dispatches_events") {
  using process_t = stateforward::sml::back::process<dummy_event>;

  owner_probe owner{};
  emel::detail::process_support<owner_probe, process_t> support{&owner};
  support.queue_.push(dummy_event{});
  CHECK(owner.calls == 1);

  emel::detail::process_support<owner_probe, process_t> no_owner{nullptr};
  no_owner.queue_.push(dummy_event{});
  CHECK(owner.calls == 1);
}

TEST_CASE("stateforward_sml_dispatch_table_routes_runtime_events") {
  namespace sml = stateforward::sml;
  sml::sm<dispatch_surface_model> machine{};
  auto dispatch_event = sml::utility::make_dispatch_table<runtime_event, 1, 2>(machine);

  int32_t marker = 0;
  CHECK(dispatch_event(runtime_event{.id = 1, .marker = marker}, 1));
  CHECK(marker == 1);
  CHECK(machine.is(sml::state<state_seen_one>));

  CHECK_FALSE(dispatch_event(runtime_event{.id = 99, .marker = marker}, 99));
  CHECK(marker == 1);
  CHECK(machine.is(sml::state<state_seen_one>));

  CHECK(dispatch_event(runtime_event{.id = 2, .marker = marker}, 2));
  CHECK(marker == 2);
  CHECK(machine.is(sml::X));
}

TEST_CASE("stateforward_sml_logger_policy_observes_dispatch") {
  namespace sml = stateforward::sml;
  logger_counters logger{};
  sml::sm<logger_surface_model, sml::logger<logger_counters>> machine{logger};

  int32_t marker = 0;
  CHECK(machine.process_event(logger_event{.marker = marker}));

  CHECK(marker == 7);
  CHECK(logger.processed >= 1);
  CHECK(logger.guards >= 1);
  CHECK(logger.actions >= 1);
  CHECK(logger.state_changes >= 1);
  CHECK(machine.is(sml::state<state_logger_done>));
}
