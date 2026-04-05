#include <boost/sml.hpp>
#include <doctest/doctest.h>
#include <type_traits>

#include "emel/batch/planner/sm.hpp"

TEST_CASE("batch_planner_public_alias_uses_canonical_planner_name") {
  static_assert(std::is_same_v<emel::BatchPlanner, emel::batch::planner::sm>);
  static_assert(std::is_same_v<emel::batch::planner::Planner, emel::batch::planner::sm>);
  static_assert(std::is_same_v<emel::batch::planner::sm::model_type,
                               emel::batch::planner::model>);
  static_assert(std::is_same_v<emel::batch::planner::sm::context_type,
                               emel::batch::planner::action::context>);

  emel::BatchPlanner machine{};
  CHECK(machine.is(boost::sml::state<emel::batch::planner::initialized>));
}
