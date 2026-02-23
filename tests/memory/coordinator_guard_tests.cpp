#include <doctest/doctest.h>

#include "emel/memory/coordinator/guards.hpp"
#include "emel/memory/coordinator/hybrid/guards.hpp"
#include "emel/memory/coordinator/kv/guards.hpp"
#include "emel/memory/coordinator/recurrent/guards.hpp"

#define EXERCISE_COORDINATOR_GUARDS(CTX_T, KIND_T, GUARD_NS, OPTIMIZE_MAKES_READY)            \
  do {                                                                                          \
    CTX_T ctx{};                                                                                \
    CHECK(GUARD_NS::phase_ok{}(ctx));                                                           \
    CHECK_FALSE(GUARD_NS::phase_failed{}(ctx));                                                 \
    ctx.phase_error = EMEL_ERR_BACKEND;                                                         \
    CHECK_FALSE(GUARD_NS::phase_ok{}(ctx));                                                     \
    CHECK(GUARD_NS::phase_failed{}(ctx));                                                       \
                                                                                                \
    ctx = {};                                                                                   \
    CHECK_FALSE(GUARD_NS::valid_update_context{}(ctx));                                         \
    CHECK(GUARD_NS::invalid_update_context{}(ctx));                                             \
    ctx.active_request = KIND_T::update;                                                        \
    CHECK(GUARD_NS::valid_update_context{}(ctx));                                               \
    CHECK_FALSE(GUARD_NS::invalid_update_context{}(ctx));                                       \
                                                                                                \
    ctx.active_request = KIND_T::update;                                                        \
    ctx.batch_request.n_ubatch = 1;                                                             \
    ctx.batch_request.n_ubatches_total = 1;                                                     \
    CHECK_FALSE(GUARD_NS::valid_batch_context{}(ctx));                                          \
    ctx.active_request = KIND_T::batch;                                                         \
    ctx.batch_request.n_ubatch = 0;                                                             \
    ctx.batch_request.n_ubatches_total = 1;                                                     \
    CHECK_FALSE(GUARD_NS::valid_batch_context{}(ctx));                                          \
    CHECK(GUARD_NS::invalid_batch_context{}(ctx));                                              \
    ctx.batch_request.n_ubatch = 1;                                                             \
    ctx.batch_request.n_ubatches_total = 0;                                                     \
    CHECK_FALSE(GUARD_NS::valid_batch_context{}(ctx));                                          \
    ctx.batch_request.n_ubatch = 1;                                                             \
    ctx.batch_request.n_ubatches_total = 2;                                                     \
    CHECK(GUARD_NS::valid_batch_context{}(ctx));                                                \
    CHECK_FALSE(GUARD_NS::invalid_batch_context{}(ctx));                                        \
                                                                                                \
    ctx.active_request = KIND_T::update;                                                        \
    CHECK_FALSE(GUARD_NS::valid_full_context{}(ctx));                                           \
    CHECK(GUARD_NS::invalid_full_context{}(ctx));                                               \
    ctx.active_request = KIND_T::full;                                                          \
    CHECK(GUARD_NS::valid_full_context{}(ctx));                                                 \
    CHECK_FALSE(GUARD_NS::invalid_full_context{}(ctx));                                         \
                                                                                                \
    ctx.prepared_status = emel::memory::coordinator::event::memory_status::success;            \
    CHECK(GUARD_NS::prepare_update_success{}(ctx));                                             \
    CHECK_FALSE(GUARD_NS::prepare_update_no_update{}(ctx));                                     \
    CHECK_FALSE(GUARD_NS::prepare_update_invalid_status{}(ctx));                                \
    ctx.prepared_status = emel::memory::coordinator::event::memory_status::no_update;          \
    CHECK_FALSE(GUARD_NS::prepare_update_success{}(ctx));                                       \
    CHECK(GUARD_NS::prepare_update_no_update{}(ctx));                                           \
    CHECK_FALSE(GUARD_NS::prepare_update_invalid_status{}(ctx));                                \
    ctx.prepared_status = emel::memory::coordinator::event::memory_status::failed_prepare;     \
    CHECK_FALSE(GUARD_NS::prepare_update_success{}(ctx));                                       \
    CHECK_FALSE(GUARD_NS::prepare_update_no_update{}(ctx));                                     \
    CHECK(GUARD_NS::prepare_update_invalid_status{}(ctx));                                      \
                                                                                                \
    ctx.active_request = KIND_T::update;                                                        \
    ctx.prepared_status = emel::memory::coordinator::event::memory_status::success;            \
    ctx.has_pending_update = false;                                                             \
    ctx.update_request.optimize = false;                                                        \
    CHECK_FALSE(GUARD_NS::apply_update_ready{}(ctx));                                           \
    CHECK(GUARD_NS::apply_update_backend_failed{}(ctx));                                        \
    CHECK_FALSE(GUARD_NS::apply_update_invalid_context{}(ctx));                                 \
                                                                                                \
    ctx.update_request.optimize = true;                                                         \
    CHECK(GUARD_NS::apply_update_ready{}(ctx) == OPTIMIZE_MAKES_READY);                        \
    CHECK(GUARD_NS::apply_update_backend_failed{}(ctx) != OPTIMIZE_MAKES_READY);               \
                                                                                                \
    ctx.has_pending_update = true;                                                              \
    ctx.update_request.optimize = false;                                                        \
    CHECK(GUARD_NS::apply_update_ready{}(ctx));                                                 \
    CHECK_FALSE(GUARD_NS::apply_update_backend_failed{}(ctx));                                  \
                                                                                                \
    ctx.prepared_status = emel::memory::coordinator::event::memory_status::no_update;          \
    CHECK_FALSE(GUARD_NS::apply_update_ready{}(ctx));                                           \
    CHECK_FALSE(GUARD_NS::apply_update_backend_failed{}(ctx));                                  \
    CHECK(GUARD_NS::apply_update_invalid_context{}(ctx));                                       \
                                                                                                \
    ctx.active_request = KIND_T::update;                                                        \
    CHECK(GUARD_NS::valid_publish_update_context{}(ctx));                                       \
    CHECK_FALSE(GUARD_NS::invalid_publish_update_context{}(ctx));                               \
    CHECK_FALSE(GUARD_NS::valid_publish_batch_context{}(ctx));                                  \
    CHECK(GUARD_NS::invalid_publish_batch_context{}(ctx));                                      \
    CHECK_FALSE(GUARD_NS::valid_publish_full_context{}(ctx));                                   \
    CHECK(GUARD_NS::invalid_publish_full_context{}(ctx));                                       \
                                                                                                \
    ctx.active_request = KIND_T::batch;                                                         \
    CHECK_FALSE(GUARD_NS::valid_publish_update_context{}(ctx));                                 \
    CHECK(GUARD_NS::invalid_publish_update_context{}(ctx));                                     \
    CHECK(GUARD_NS::valid_publish_batch_context{}(ctx));                                        \
    CHECK_FALSE(GUARD_NS::invalid_publish_batch_context{}(ctx));                                \
    CHECK_FALSE(GUARD_NS::valid_publish_full_context{}(ctx));                                   \
    CHECK(GUARD_NS::invalid_publish_full_context{}(ctx));                                       \
                                                                                                \
    ctx.active_request = KIND_T::full;                                                          \
    CHECK_FALSE(GUARD_NS::valid_publish_update_context{}(ctx));                                 \
    CHECK(GUARD_NS::invalid_publish_update_context{}(ctx));                                     \
    CHECK_FALSE(GUARD_NS::valid_publish_batch_context{}(ctx));                                  \
    CHECK(GUARD_NS::invalid_publish_batch_context{}(ctx));                                      \
    CHECK(GUARD_NS::valid_publish_full_context{}(ctx));                                         \
    CHECK_FALSE(GUARD_NS::invalid_publish_full_context{}(ctx));                                 \
  } while (false)

TEST_CASE("memory_coordinator_guard_predicates_cover_all_branches") {
  EXERCISE_COORDINATOR_GUARDS(
      emel::memory::coordinator::action::context,
      emel::memory::coordinator::action::request_kind,
      emel::memory::coordinator::guard,
      true);
}

TEST_CASE("memory_coordinator_hybrid_guard_predicates_cover_all_branches") {
  EXERCISE_COORDINATOR_GUARDS(
      emel::memory::coordinator::hybrid::action::context,
      emel::memory::coordinator::hybrid::action::request_kind,
      emel::memory::coordinator::hybrid::guard,
      true);
}

TEST_CASE("memory_coordinator_kv_guard_predicates_cover_all_branches") {
  EXERCISE_COORDINATOR_GUARDS(
      emel::memory::coordinator::kv::action::context,
      emel::memory::coordinator::kv::action::request_kind,
      emel::memory::coordinator::kv::guard,
      true);
}

TEST_CASE("memory_coordinator_recurrent_guard_predicates_cover_all_branches") {
  EXERCISE_COORDINATOR_GUARDS(
      emel::memory::coordinator::recurrent::action::context,
      emel::memory::coordinator::recurrent::action::request_kind,
      emel::memory::coordinator::recurrent::guard,
      false);
}

#undef EXERCISE_COORDINATOR_GUARDS
