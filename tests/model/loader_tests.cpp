#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <boost/sml.hpp>

#include "emel/model/loader/sm.hpp"
#include "emel/model/loader/guards.hpp"

namespace sml = boost::sml;

TEST_CASE("loader_sm_starts_in_initialized") {
  emel::model::loader::sm machine = {};
  int state_count = 0;
  machine.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("loader_sm_transitions_on_start") {
  emel::model::loader::sm machine = {};
  emel::model::data model_data = {};
  CHECK(machine.process_event(emel::model::loader::event::load{.model_data = model_data}));
}

TEST_CASE("loader_guards_report_expected_results") {
  using namespace emel::model::loader;

  CHECK(guard::no_error());
  CHECK_FALSE(guard::error());
  CHECK_FALSE(guard::has_error());
  CHECK_FALSE(guard::has_arch_validate());
  CHECK_FALSE(guard::no_error_and_has_arch_validate());
  CHECK(guard::no_error_and_no_arch_validate());
}
