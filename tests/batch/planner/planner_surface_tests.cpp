#include <boost/sml.hpp>
#include <array>
#include <doctest/doctest.h>
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

TEST_CASE("batch_planner_mode_wrappers_do_not_reuse_stale_outcomes_after_completion") {
  planner_done_capture planner_done{};
  planner_error_capture planner_error{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  const emel::batch::planner::event::plan_request request{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_steps = 2,
    .mode = emel::batch::planner::event::plan_mode::simple,
    .output_all = true,
    .on_done = make_planner_done(&planner_done),
    .on_error = make_planner_error(&planner_error),
  };
  const emel::error::type internal_error =
      emel::error::cast(emel::batch::planner::error::internal_error);

  SUBCASE("simple") {
    emel::batch::planner::event::plan_scratch first_ctx{};
    first_ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::simple::events::plan_done,
                 emel::batch::planner::modes::simple::events::plan_error>
        first{};
    emel::batch::planner::modes::simple::sm mode{};
    CHECK(mode.process_event(emel::batch::planner::modes::simple::event::plan_request{
      .request = request,
      .ctx = first_ctx,
      .on_done = make_mode_done(&first),
      .on_error = make_mode_error(&first),
    }));
    CHECK(first.done_called);
    CHECK_FALSE(first.error_called);

    emel::batch::planner::event::plan_scratch second_ctx{};
    second_ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::simple::events::plan_done,
                 emel::batch::planner::modes::simple::events::plan_error>
        second{};
    CHECK_FALSE(mode.process_event(emel::batch::planner::modes::simple::event::plan_request{
      .request = request,
      .ctx = second_ctx,
      .on_done = make_mode_done(&second),
      .on_error = make_mode_error(&second),
    }));
    CHECK_FALSE(second.done_called);
    CHECK(second.error_called);
    CHECK(second.err == internal_error);
  }

  SUBCASE("equal") {
    emel::batch::planner::event::plan_scratch first_ctx{};
    first_ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::equal::events::plan_done,
                 emel::batch::planner::modes::equal::events::plan_error>
        first{};
    emel::batch::planner::modes::equal::sm mode{};
    CHECK(mode.process_event(emel::batch::planner::modes::equal::event::plan_request{
      .request = request,
      .ctx = first_ctx,
      .on_done = make_mode_done(&first),
      .on_error = make_mode_error(&first),
    }));
    CHECK(first.done_called);
    CHECK_FALSE(first.error_called);

    emel::batch::planner::event::plan_scratch second_ctx{};
    second_ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::equal::events::plan_done,
                 emel::batch::planner::modes::equal::events::plan_error>
        second{};
    CHECK_FALSE(mode.process_event(emel::batch::planner::modes::equal::event::plan_request{
      .request = request,
      .ctx = second_ctx,
      .on_done = make_mode_done(&second),
      .on_error = make_mode_error(&second),
    }));
    CHECK_FALSE(second.done_called);
    CHECK(second.error_called);
    CHECK(second.err == internal_error);
  }

  SUBCASE("sequential") {
    emel::batch::planner::event::plan_scratch first_ctx{};
    first_ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::sequential::events::plan_done,
                 emel::batch::planner::modes::sequential::events::plan_error>
        first{};
    emel::batch::planner::modes::sequential::sm mode{};
    CHECK(mode.process_event(emel::batch::planner::modes::sequential::event::plan_request{
      .request = request,
      .ctx = first_ctx,
      .on_done = make_mode_done(&first),
      .on_error = make_mode_error(&first),
    }));
    CHECK(first.done_called);
    CHECK_FALSE(first.error_called);

    emel::batch::planner::event::plan_scratch second_ctx{};
    second_ctx.effective_step_size = 2;
    mode_capture<emel::batch::planner::modes::sequential::events::plan_done,
                 emel::batch::planner::modes::sequential::events::plan_error>
        second{};
    CHECK_FALSE(mode.process_event(
        emel::batch::planner::modes::sequential::event::plan_request{
          .request = request,
          .ctx = second_ctx,
          .on_done = make_mode_done(&second),
          .on_error = make_mode_error(&second),
        }));
    CHECK_FALSE(second.done_called);
    CHECK(second.error_called);
    CHECK(second.err == internal_error);
  }
}

TEST_CASE("batch_planner_surface_exports_canonical_prefixed_symbols") {
  [[maybe_unused]] const auto * effect_begin_planning =
      &emel::batch::planner::action::effect_begin_planning;
  [[maybe_unused]] const auto * effect_normalize_step_size =
      &emel::batch::planner::action::effect_normalize_step_size;
  [[maybe_unused]] const auto * effect_plan_simple_mode =
      &emel::batch::planner::action::effect_plan_simple_mode;
  [[maybe_unused]] const auto * effect_plan_equal_mode =
      &emel::batch::planner::action::effect_plan_equal_mode;
  [[maybe_unused]] const auto * effect_plan_sequential_mode =
      &emel::batch::planner::action::effect_plan_sequential_mode;
  [[maybe_unused]] const auto * effect_reject_unexpected_event =
      &emel::batch::planner::action::effect_reject_unexpected_event;

  [[maybe_unused]] const auto * guard_inputs_valid =
      &emel::batch::planner::guard::guard_inputs_valid;
  [[maybe_unused]] const auto * guard_mode_is_simple =
      &emel::batch::planner::guard::guard_mode_is_simple;
  [[maybe_unused]] const auto * guard_mode_is_equal =
      &emel::batch::planner::guard::guard_mode_is_equal;
  [[maybe_unused]] const auto * guard_mode_is_sequential =
      &emel::batch::planner::guard::guard_mode_is_sequential;
  [[maybe_unused]] const auto * guard_planning_failed_with_error =
      &emel::batch::planner::guard::guard_planning_failed_with_error;

  static_assert(std::is_default_constructible_v<emel::batch::planner::state_idle>);
  static_assert(std::is_default_constructible_v<emel::batch::planner::state_input_validation>);
  static_assert(std::is_default_constructible_v<emel::batch::planner::state_mode_selection>);
  static_assert(std::is_default_constructible_v<emel::batch::planner::state_simple_planning>);
  static_assert(std::is_default_constructible_v<emel::batch::planner::state_equal_planning>);
  static_assert(std::is_default_constructible_v<emel::batch::planner::state_sequential_planning>);
  static_assert(std::is_default_constructible_v<emel::batch::planner::state_result_publish>);
  static_assert(std::is_default_constructible_v<emel::batch::planner::state_completed>);
  static_assert(std::is_default_constructible_v<emel::batch::planner::state_request_rejected>);

  CHECK(true);
}

TEST_CASE("batch_planner_mode_actions_export_effect_symbols") {
  [[maybe_unused]] const auto * simple_begin =
      &emel::batch::planner::modes::simple::action::effect_begin_planning;
  [[maybe_unused]] const auto * simple_plan =
      &emel::batch::planner::modes::simple::action::effect_plan_simple_batches;
  [[maybe_unused]] const auto * simple_reject_invalid_step_size =
      &emel::batch::planner::modes::simple::action::effect_reject_invalid_step_size;

  [[maybe_unused]] const auto * sequential_begin =
      &emel::batch::planner::modes::sequential::action::effect_begin_planning;
  [[maybe_unused]] const auto * sequential_plan =
      &emel::batch::planner::modes::sequential::action::effect_plan_sequential_batches;
  [[maybe_unused]] const auto * sequential_reject_invalid_step_size =
      &emel::batch::planner::modes::sequential::action::effect_reject_invalid_step_size;

  [[maybe_unused]] const auto * equal_begin =
      &emel::batch::planner::modes::equal::action::effect_begin_planning;
  [[maybe_unused]] const auto * equal_plan =
      &emel::batch::planner::modes::equal::action::effect_plan_equal_batches;
  [[maybe_unused]] const auto * equal_plan_fast =
      &emel::batch::planner::modes::equal::action::effect_plan_equal_primary_batches;
  [[maybe_unused]] const auto * equal_reject_invalid_sequence_id =
      &emel::batch::planner::modes::equal::action::effect_reject_invalid_sequence_id;

  CHECK(true);
}
