#include <doctest/doctest.h>

#include "emel/emel.h"
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

}  // namespace

TEST_CASE("sm_normalize_event_result_handles_error_out") {
  int32_t err = EMEL_OK;
  dummy_event ok{.error_out = &err};
  CHECK_FALSE(emel::detail::normalize_event_result(ok, false));
  CHECK(emel::detail::normalize_event_result(ok, true));

  err = EMEL_ERR_BACKEND;
  CHECK_FALSE(emel::detail::normalize_event_result(ok, true));

  struct no_error_event {};
  CHECK(emel::detail::normalize_event_result(no_error_event{}, true));
}

TEST_CASE("sm_process_support_dispatches_events") {
  using process_t = boost::sml::back::process<dummy_event>;

  owner_probe owner{};
  emel::detail::process_support<owner_probe, process_t> support{&owner};
  support.queue_.push(dummy_event{});
  CHECK(owner.calls == 1);

  emel::detail::process_support<owner_probe, process_t> no_owner{nullptr};
  no_owner.queue_.push(dummy_event{});
  CHECK(owner.calls == 1);
}
