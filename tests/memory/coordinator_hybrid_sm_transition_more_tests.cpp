#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/hybrid/sm.hpp"

namespace {

TEST_CASE("memory_coordinator_sm_update_success_path_updates_counts") {
  emel::memory::coordinator::hybrid::action::context ctx{};
  boost::sml::sm<
    emel::memory::coordinator::hybrid::model,
    boost::sml::testing>
    machine{ctx};
  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = true,
  }));
  CHECK(ctx.prepared_status == emel::memory::coordinator::event::memory_status::success);
  CHECK(ctx.update_apply_count == 1);
  CHECK_FALSE(ctx.has_pending_update);
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::hybrid::initialized>));
}

TEST_CASE("memory_coordinator_sm_update_no_update_skips_apply") {
  emel::memory::coordinator::hybrid::action::context ctx{};
  boost::sml::sm<
    emel::memory::coordinator::hybrid::model,
    boost::sml::testing>
    machine{ctx};
  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
    .optimize = false,
  }));
  CHECK(ctx.prepared_status == emel::memory::coordinator::event::memory_status::no_update);
  CHECK(ctx.update_apply_count == 0);
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::hybrid::initialized>));
}

TEST_CASE("memory_coordinator_sm_batch_prepare_sets_pending") {
  emel::memory::coordinator::hybrid::action::context ctx{};
  boost::sml::sm<
    emel::memory::coordinator::hybrid::model,
    boost::sml::testing>
    machine{ctx};
  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_batch{
    .n_ubatch = 1,
    .n_ubatches_total = 2,
  }));
  CHECK(ctx.prepared_status == emel::memory::coordinator::event::memory_status::success);
  CHECK(ctx.batch_prepare_count == 1);
  CHECK(ctx.has_pending_update);
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::hybrid::initialized>));
}

TEST_CASE("memory_coordinator_sm_full_prepare_sets_pending") {
  emel::memory::coordinator::hybrid::action::context ctx{};
  boost::sml::sm<
    emel::memory::coordinator::hybrid::model,
    boost::sml::testing>
    machine{ctx};
  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_full{
  }));
  CHECK(ctx.prepared_status == emel::memory::coordinator::event::memory_status::success);
  CHECK(ctx.full_prepare_count == 1);
  CHECK(ctx.has_pending_update);
  CHECK(machine.is(boost::sml::state<emel::memory::coordinator::hybrid::initialized>));
}

}  // namespace
