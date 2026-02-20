#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/hybrid/actions.hpp"
#include "emel/memory/coordinator/hybrid/guards.hpp"

TEST_CASE("memory_coordinator_on_unexpected_sets_backend_error") {
  emel::memory::coordinator::hybrid::action::context ctx{};
  emel::memory::coordinator::hybrid::action::on_unexpected(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}

TEST_CASE("memory_coordinator_prepare_update_invalid_status_guard") {
  emel::memory::coordinator::hybrid::action::context ctx{};
  ctx.active_request = emel::memory::coordinator::hybrid::action::request_kind::update;
  ctx.prepared_status = emel::memory::coordinator::event::memory_status::failed_compute;

  CHECK(emel::memory::coordinator::hybrid::guard::prepare_update_invalid_status{}(ctx));
  CHECK_FALSE(emel::memory::coordinator::hybrid::guard::prepare_update_success{}(ctx));
  CHECK_FALSE(emel::memory::coordinator::hybrid::guard::prepare_update_no_update{}(ctx));
}
