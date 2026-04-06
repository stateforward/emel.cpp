#include <boost/sml.hpp>
#include <array>
#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>

#include "emel/batch/planner/modes/equal/context.hpp"
#include "emel/batch/planner/modes/equal/errors.hpp"
#include "emel/batch/planner/modes/equal/events.hpp"
#include "emel/batch/planner/modes/sequential/context.hpp"
#include "emel/batch/planner/modes/sequential/errors.hpp"
#include "emel/batch/planner/modes/sequential/events.hpp"
#include "emel/batch/planner/modes/simple/context.hpp"
#include "emel/batch/planner/modes/simple/errors.hpp"
#include "emel/batch/planner/modes/simple/events.hpp"
#include "emel/batch/planner/sm.hpp"
#include "emel/callback.hpp"

namespace {

struct planner_done_capture {
  void on_done(const emel::batch::planner::events::plan_done &) noexcept {}
};

struct planner_error_capture {
  void on_error(const emel::batch::planner::events::plan_error &) noexcept {}
};

template <class done_event, class error_event>
struct mode_capture {
  bool done_called = false;
  bool error_called = false;
  emel::error::type err = emel::error::type{};

  void on_done(const done_event &) noexcept { done_called = true; }

  void on_error(const error_event & ev) noexcept {
    error_called = true;
    err = ev.err;
  }
};

inline emel::callback<void(const emel::batch::planner::events::plan_done &)> make_planner_done(
    planner_done_capture * capture) {
  return emel::callback<void(const emel::batch::planner::events::plan_done &)>::from<
      planner_done_capture,
      &planner_done_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::planner::events::plan_error &)> make_planner_error(
    planner_error_capture * capture) {
  return emel::callback<void(const emel::batch::planner::events::plan_error &)>::from<
      planner_error_capture,
      &planner_error_capture::on_error>(capture);
}

template <class done_event, class error_event>
inline emel::callback<void(const done_event &)> make_mode_done(
    mode_capture<done_event, error_event> * capture) {
  return emel::callback<void(const done_event &)>::template from<
      mode_capture<done_event, error_event>,
      &mode_capture<done_event, error_event>::on_done>(capture);
}

template <class done_event, class error_event>
inline emel::callback<void(const error_event &)> make_mode_error(
    mode_capture<done_event, error_event> * capture) {
  return emel::callback<void(const error_event &)>::template from<
      mode_capture<done_event, error_event>,
      &mode_capture<done_event, error_event>::on_error>(capture);
}

inline std::string load_text(const std::filesystem::path & path) {
  std::ifstream input(path);
  REQUIRE(input.good());
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

inline std::filesystem::path find_repo_root(std::filesystem::path path) {
  while (!path.empty()) {
    if (std::filesystem::exists(path / "CMakeLists.txt")) {
      return path;
    }
    path = path.parent_path();
  }
  return {};
}

inline std::string process_event_block(const std::filesystem::path & relative_path) {
  const auto repo_root = find_repo_root(std::filesystem::path(__FILE__).parent_path());
  REQUIRE_FALSE(repo_root.empty());
  const std::string source = load_text(repo_root / relative_path);
  const auto process_event_pos = source.find("bool process_event(const event::plan_request & ev) {");
  REQUIRE(process_event_pos != std::string::npos);
  const auto namespace_pos = source.find("}  // namespace", process_event_pos);
  REQUIRE(namespace_pos != std::string::npos);
  return source.substr(process_event_pos, namespace_pos - process_event_pos);
}

inline std::string load_repo_text(const std::filesystem::path & relative_path) {
  const auto repo_root = find_repo_root(std::filesystem::path(__FILE__).parent_path());
  REQUIRE_FALSE(repo_root.empty());
  return load_text(repo_root / relative_path);
}

}  // namespace

TEST_CASE("batch_planner_public_alias_uses_canonical_planner_name") {
  static_assert(std::is_same_v<emel::BatchPlanner, emel::batch::planner::sm>);
  static_assert(std::is_same_v<emel::batch::planner::Planner, emel::batch::planner::sm>);
  static_assert(std::is_same_v<emel::batch::planner::sm::model_type,
                               emel::batch::planner::model>);
  static_assert(std::is_same_v<emel::batch::planner::sm::context_type,
                               emel::batch::planner::action::context>);

  emel::BatchPlanner machine{};
  CHECK(machine.is(boost::sml::state<emel::batch::planner::state_idle>));
}

TEST_CASE("batch_planner_mode_surfaces_export_canonical_mode_aliases") {
  static_assert(std::is_same_v<emel::batch::planner::modes::simple::context,
                               emel::batch::planner::action::context>);
  static_assert(std::is_same_v<emel::batch::planner::modes::simple::error,
                               emel::batch::planner::error>);
  static_assert(std::is_same_v<
                decltype(emel::batch::planner::modes::simple::event::plan_request::request),
                const emel::batch::planner::event::plan_request &>);
  static_assert(std::is_same_v<
                decltype(emel::batch::planner::modes::simple::event::plan_request::ctx),
                emel::batch::planner::event::plan_scratch &>);

  static_assert(std::is_same_v<emel::batch::planner::modes::sequential::context,
                               emel::batch::planner::action::context>);
  static_assert(std::is_same_v<emel::batch::planner::modes::sequential::error,
                               emel::batch::planner::error>);
  static_assert(std::is_same_v<
                decltype(emel::batch::planner::modes::sequential::event::plan_request::request),
                const emel::batch::planner::event::plan_request &>);
  static_assert(std::is_same_v<
                decltype(emel::batch::planner::modes::sequential::event::plan_request::ctx),
                emel::batch::planner::event::plan_scratch &>);

  static_assert(std::is_same_v<emel::batch::planner::modes::equal::context,
                               emel::batch::planner::action::context>);
  static_assert(std::is_same_v<emel::batch::planner::modes::equal::error,
                               emel::batch::planner::error>);
  static_assert(std::is_same_v<
                decltype(emel::batch::planner::modes::equal::event::plan_request::request),
                const emel::batch::planner::event::plan_request &>);
  static_assert(std::is_same_v<
                decltype(emel::batch::planner::modes::equal::event::plan_request::ctx),
                emel::batch::planner::event::plan_scratch &>);

  CHECK(true);
}

TEST_CASE("batch_planner_mode_wrappers_emit_typed_outcome_events") {
  planner_done_capture planner_done{};
  planner_error_capture planner_error{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_steps = 2,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .output_all = true,
    .on_done = make_planner_done(&planner_done),
    .on_error = make_planner_error(&planner_error),
  };

  SUBCASE("simple") {
    emel::batch::planner::event::plan_scratch ctx{};
    ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::simple::events::plan_done,
                 emel::batch::planner::modes::simple::events::plan_error>
        capture{};
    emel::batch::planner::modes::simple::sm mode{};
    CHECK(mode.process_event(emel::batch::planner::modes::simple::event::plan_request{
      .request = request,
      .ctx = ctx,
      .on_done = make_mode_done(&capture),
      .on_error = make_mode_error(&capture),
    }));
    CHECK(capture.done_called);
    CHECK_FALSE(capture.error_called);
  }

  SUBCASE("equal") {
    emel::batch::planner::event::plan_scratch ctx{};
    ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::equal::events::plan_done,
                 emel::batch::planner::modes::equal::events::plan_error>
        capture{};
    emel::batch::planner::modes::equal::sm mode{};
    CHECK(mode.process_event(emel::batch::planner::modes::equal::event::plan_request{
      .request = request,
      .ctx = ctx,
      .on_done = make_mode_done(&capture),
      .on_error = make_mode_error(&capture),
    }));
    CHECK(capture.done_called);
    CHECK_FALSE(capture.error_called);
  }

  SUBCASE("sequential") {
    emel::batch::planner::event::plan_scratch ctx{};
    ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::sequential::events::plan_done,
                 emel::batch::planner::modes::sequential::events::plan_error>
        capture{};
    emel::batch::planner::modes::sequential::sm mode{};
    CHECK(mode.process_event(emel::batch::planner::modes::sequential::event::plan_request{
      .request = request,
      .ctx = ctx,
      .on_done = make_mode_done(&capture),
      .on_error = make_mode_error(&capture),
    }));
    CHECK(capture.done_called);
    CHECK_FALSE(capture.error_called);
  }
}

TEST_CASE("batch_planner_mode_wrappers_use_state_inspection_for_outcome_dispatch") {
  const std::array<std::filesystem::path, 3> wrapper_paths = {{
      "src/emel/batch/planner/modes/simple/sm.hpp",
      "src/emel/batch/planner/modes/equal/sm.hpp",
      "src/emel/batch/planner/modes/sequential/sm.hpp",
  }};

  for (const auto & wrapper_path : wrapper_paths) {
    const std::string block = process_event_block(wrapper_path);
    CHECK(block.find("if (") == std::string::npos);
    CHECK(block.find("guard_planning_succeeded") == std::string::npos);
    CHECK(block.find("detail::complete_mode_request") != std::string::npos);
  }
}

TEST_CASE("batch_planner_surface_uses_planner_family_prefix_conventions") {
  const std::string events_source = load_repo_text("src/emel/batch/planner/events.hpp");
  const std::string actions_source = load_repo_text("src/emel/batch/planner/actions.hpp");
  const std::string guards_source = load_repo_text("src/emel/batch/planner/guards.hpp");
  const std::string sm_source = load_repo_text("src/emel/batch/planner/sm.hpp");

  CHECK(events_source.find("struct plan_request") != std::string::npos);
  CHECK(events_source.find("struct plan_scratch") != std::string::npos);
  CHECK(events_source.find("struct plan_runtime") != std::string::npos);
  CHECK(events_source.find("struct request {") == std::string::npos);
  CHECK(events_source.find("struct request_ctx") == std::string::npos);
  CHECK(events_source.find("struct request_runtime") == std::string::npos);

  CHECK(actions_source.find("inline constexpr auto effect_begin_planning") != std::string::npos);
  CHECK(actions_source.find("inline constexpr auto effect_normalize_step_size") != std::string::npos);
  CHECK(actions_source.find("inline constexpr auto effect_plan_simple_mode") != std::string::npos);
  CHECK(actions_source.find("inline constexpr auto effect_plan_equal_mode") != std::string::npos);
  CHECK(actions_source.find("inline constexpr auto effect_plan_sequential_mode") != std::string::npos);
  CHECK(actions_source.find("inline constexpr auto effect_reject_unexpected_event") !=
        std::string::npos);
  CHECK(actions_source.find("inline constexpr auto begin_planning =") == std::string::npos);
  CHECK(actions_source.find("inline constexpr auto normalize_step_size =") == std::string::npos);
  CHECK(actions_source.find("inline constexpr auto plan_simple_mode =") == std::string::npos);
  CHECK(actions_source.find("inline constexpr auto plan_equal_mode =") == std::string::npos);
  CHECK(actions_source.find("inline constexpr auto plan_sequential_mode =") ==
        std::string::npos);
  CHECK(actions_source.find("inline constexpr auto reject_unexpected_event =") ==
        std::string::npos);

  CHECK(guards_source.find("inline constexpr auto guard_inputs_valid") != std::string::npos);
  CHECK(guards_source.find("inline constexpr auto guard_mode_is_simple") != std::string::npos);
  CHECK(guards_source.find("inline constexpr auto guard_mode_is_equal") != std::string::npos);
  CHECK(guards_source.find("inline constexpr auto guard_mode_is_sequential") !=
        std::string::npos);
  CHECK(guards_source.find("inline constexpr auto guard_planning_failed_with_error") !=
        std::string::npos);
  CHECK(guards_source.find("inline constexpr auto inputs_valid =") == std::string::npos);
  CHECK(guards_source.find("inline constexpr auto mode_is_simple =") == std::string::npos);
  CHECK(guards_source.find("inline constexpr auto mode_is_equal =") == std::string::npos);
  CHECK(guards_source.find("inline constexpr auto mode_is_sequential =") == std::string::npos);
  CHECK(guards_source.find("inline constexpr auto planning_failed_with_error =") ==
        std::string::npos);

  CHECK(sm_source.find("state_idle") != std::string::npos);
  CHECK(sm_source.find("state_input_validation") != std::string::npos);
  CHECK(sm_source.find("state_mode_selection") != std::string::npos);
  CHECK(sm_source.find("state_simple_planning") != std::string::npos);
  CHECK(sm_source.find("state_equal_planning") != std::string::npos);
  CHECK(sm_source.find("state_sequential_planning") != std::string::npos);
  CHECK(sm_source.find("state_result_publish") != std::string::npos);
  CHECK(sm_source.find("state_completed") != std::string::npos);
  CHECK(sm_source.find("state_request_rejected") != std::string::npos);
  CHECK(sm_source.find("state_initialized") == std::string::npos);
  CHECK(sm_source.find("state_validate_decision") == std::string::npos);
  CHECK(sm_source.find("state_mode_decision") == std::string::npos);
}

TEST_CASE("batch_planner_mode_actions_use_effect_prefix_without_wrapper_trampolines") {
  const std::string simple_actions =
      load_repo_text("src/emel/batch/planner/modes/simple/actions.hpp");
  const std::string equal_actions =
      load_repo_text("src/emel/batch/planner/modes/equal/actions.hpp");
  const std::string sequential_actions =
      load_repo_text("src/emel/batch/planner/modes/sequential/actions.hpp");

  CHECK(simple_actions.find("inline constexpr auto effect_begin_planning") != std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto effect_reject_invalid_step_size") !=
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto effect_reject_output_steps_full") !=
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto effect_reject_output_indices_full") !=
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto effect_reject_planning_progress_stalled") !=
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto effect_plan_simple_batches") !=
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto begin_planning =") == std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto reject_invalid_step_size =") ==
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto reject_output_steps_full =") ==
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto reject_output_indices_full =") ==
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto reject_planning_progress_stalled =") ==
        std::string::npos);
  CHECK(simple_actions.find("inline constexpr auto plan_simple_batches =") ==
        std::string::npos);
  CHECK(simple_actions.find("_impl(") == std::string::npos);

  CHECK(equal_actions.find("inline constexpr auto effect_begin_planning") != std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto effect_reject_invalid_step_size") !=
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto effect_reject_invalid_sequence_id") !=
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto effect_reject_output_steps_full") !=
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto effect_reject_output_indices_full") !=
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto effect_reject_planning_progress_stalled") !=
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto effect_plan_equal_batches") !=
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto effect_plan_equal_primary_batches") !=
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto begin_planning =") == std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto reject_invalid_step_size =") ==
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto reject_invalid_sequence_id =") ==
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto reject_output_steps_full =") ==
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto reject_output_indices_full =") ==
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto reject_planning_progress_stalled =") ==
        std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto plan_equal_batches =") == std::string::npos);
  CHECK(equal_actions.find("inline constexpr auto plan_equal_primary_batches =") ==
        std::string::npos);
  CHECK(equal_actions.find("_impl(") == std::string::npos);

  CHECK(sequential_actions.find("inline constexpr auto effect_begin_planning") !=
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto effect_reject_invalid_step_size") !=
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto effect_reject_output_steps_full") !=
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto effect_reject_output_indices_full") !=
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto effect_reject_planning_progress_stalled") !=
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto effect_plan_sequential_batches") !=
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto begin_planning =") == std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto reject_invalid_step_size =") ==
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto reject_output_steps_full =") ==
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto reject_output_indices_full =") ==
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto reject_planning_progress_stalled =") ==
        std::string::npos);
  CHECK(sequential_actions.find("inline constexpr auto plan_sequential_batches =") ==
        std::string::npos);
  CHECK(sequential_actions.find("_impl(") == std::string::npos);
}

TEST_CASE("batch_planner_transient_mode_states_define_unexpected_event_handlers") {
  const std::string source = load_repo_text("src/emel/batch/planner/sm.hpp");

  CHECK(source.find("sml::state<state_idle> <= sml::state<state_simple_planning>\n"
                    "          + sml::unexpected_event<sml::_>") != std::string::npos);
  CHECK(source.find("sml::state<state_idle> <= sml::state<state_equal_planning>\n"
                    "          + sml::unexpected_event<sml::_>") != std::string::npos);
  CHECK(source.find("sml::state<state_idle> <= sml::state<state_sequential_planning>\n"
                    "          + sml::unexpected_event<sml::_>") != std::string::npos);
}
