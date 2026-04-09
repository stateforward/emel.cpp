#pragma once

#include "emel/batch/planner/events.hpp"

namespace emel::batch::planner::modes::sequential {

namespace planner_event = emel::batch::planner::event;

namespace events {

struct plan_done;
struct plan_error;

}  // namespace events

namespace event {

struct plan_request {
  const planner_event::plan_request & request;
  planner_event::plan_scratch & ctx;
  const emel::callback<void(const events::plan_done &)> & on_done;
  const emel::callback<void(const events::plan_error &)> & on_error;
};

using plan_runtime = plan_request;

}  // namespace event

namespace events {

struct plan_done {
  const event::plan_request & request;
};

struct plan_error {
  const event::plan_request & request;
  emel::error::type err = emel::error::type{};
};

}  // namespace events

}  // namespace emel::batch::planner::modes::sequential
