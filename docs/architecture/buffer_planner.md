# buffer_planner

Source: `emel/buffer/planner/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> resetting : emel::buffer::planner::event::plan [boost::sml::aux::zero_wrapper<emel::buffer::planner::guard::valid_plan>] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::begin_plan>
  idle --> errored : emel::buffer::planner::event::plan [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::reject_plan>
  resetting --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:75:11)>
  resetting --> seeding_leafs : emel::buffer::planner::event::reset_done [boost::sml::front::always] / boost::sml::front::none
  resetting --> errored : emel::buffer::planner::event::reset_error [boost::sml::front::always] / boost::sml::front::none
  seeding_leafs --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:94:11)>
  seeding_leafs --> counting_references : emel::buffer::planner::event::seed_leafs_done [boost::sml::front::always] / boost::sml::front::none
  seeding_leafs --> errored : emel::buffer::planner::event::seed_leafs_error [boost::sml::front::always] / boost::sml::front::none
  counting_references --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:116:11)>
  counting_references --> allocating_explicit_inputs : emel::buffer::planner::event::count_references_done [boost::sml::front::always] / boost::sml::front::none
  counting_references --> errored : emel::buffer::planner::event::count_references_error [boost::sml::front::always] / boost::sml::front::none
  allocating_explicit_inputs --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:139:11)>
  allocating_explicit_inputs --> planning_nodes : emel::buffer::planner::event::alloc_explicit_inputs_done [boost::sml::front::always] / boost::sml::front::none
  allocating_explicit_inputs --> errored : emel::buffer::planner::event::alloc_explicit_inputs_error [boost::sml::front::always] / boost::sml::front::none
  planning_nodes --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:163:11)>
  planning_nodes --> releasing_expired : emel::buffer::planner::event::plan_nodes_done [boost::sml::front::always] / boost::sml::front::none
  planning_nodes --> errored : emel::buffer::planner::event::plan_nodes_error [boost::sml::front::always] / boost::sml::front::none
  releasing_expired --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:187:11)>
  releasing_expired --> finalizing : emel::buffer::planner::event::release_expired_done [boost::sml::front::always] / boost::sml::front::none
  releasing_expired --> errored : emel::buffer::planner::event::release_expired_error [boost::sml::front::always] / boost::sml::front::none
  finalizing --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:211:11)>
  finalizing --> splitting_required : emel::buffer::planner::event::finalize_done [boost::sml::front::always] / boost::sml::front::none
  finalizing --> errored : emel::buffer::planner::event::finalize_error [boost::sml::front::always] / boost::sml::front::none
  splitting_required --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:234:11)>
  splitting_required --> done : emel::buffer::planner::event::split_required_done [boost::sml::front::always] / boost::sml::front::none
  splitting_required --> errored : emel::buffer::planner::event::split_required_error [boost::sml::front::always] / boost::sml::front::none
  done --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:259:11)>
  done --> idle : emel::buffer::planner::events::plan_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_plan_done>
  errored --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:275:11)>
  errored --> idle : emel::buffer::planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_plan_error>
  idle --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  resetting --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  seeding_leafs --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  counting_references --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  allocating_explicit_inputs --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  planning_nodes --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  releasing_expired --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  finalizing --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  splitting_required --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  done --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
  errored --> errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `idle` | `emel::buffer::planner::event::plan` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::guard::valid_plan>` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::begin_plan>` | `resetting` |
| `idle` | `emel::buffer::planner::event::plan` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::reject_plan>` | `errored` |
| `resetting` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:75:11)>` | `boost::sml::front::internal` |
| `resetting` | `emel::buffer::planner::event::reset_done` | `boost::sml::front::always` | `boost::sml::front::none` | `seeding_leafs` |
| `resetting` | `emel::buffer::planner::event::reset_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `seeding_leafs` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:94:11)>` | `boost::sml::front::internal` |
| `seeding_leafs` | `emel::buffer::planner::event::seed_leafs_done` | `boost::sml::front::always` | `boost::sml::front::none` | `counting_references` |
| `seeding_leafs` | `emel::buffer::planner::event::seed_leafs_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `counting_references` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:116:11)>` | `boost::sml::front::internal` |
| `counting_references` | `emel::buffer::planner::event::count_references_done` | `boost::sml::front::always` | `boost::sml::front::none` | `allocating_explicit_inputs` |
| `counting_references` | `emel::buffer::planner::event::count_references_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `allocating_explicit_inputs` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:139:11)>` | `boost::sml::front::internal` |
| `allocating_explicit_inputs` | `emel::buffer::planner::event::alloc_explicit_inputs_done` | `boost::sml::front::always` | `boost::sml::front::none` | `planning_nodes` |
| `allocating_explicit_inputs` | `emel::buffer::planner::event::alloc_explicit_inputs_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `planning_nodes` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:163:11)>` | `boost::sml::front::internal` |
| `planning_nodes` | `emel::buffer::planner::event::plan_nodes_done` | `boost::sml::front::always` | `boost::sml::front::none` | `releasing_expired` |
| `planning_nodes` | `emel::buffer::planner::event::plan_nodes_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `releasing_expired` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:187:11)>` | `boost::sml::front::internal` |
| `releasing_expired` | `emel::buffer::planner::event::release_expired_done` | `boost::sml::front::always` | `boost::sml::front::none` | `finalizing` |
| `releasing_expired` | `emel::buffer::planner::event::release_expired_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `finalizing` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:211:11)>` | `boost::sml::front::internal` |
| `finalizing` | `emel::buffer::planner::event::finalize_done` | `boost::sml::front::always` | `boost::sml::front::none` | `splitting_required` |
| `finalizing` | `emel::buffer::planner::event::finalize_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `splitting_required` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:234:11)>` | `boost::sml::front::internal` |
| `splitting_required` | `emel::buffer::planner::event::split_required_done` | `boost::sml::front::always` | `boost::sml::front::none` | `done` |
| `splitting_required` | `emel::buffer::planner::event::split_required_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `done` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:259:11)>` | `boost::sml::front::internal` |
| `done` | `emel::buffer::planner::events::plan_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_plan_done>` | `idle` |
| `errored` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/planner/sm.hpp:275:11)>` | `boost::sml::front::internal` |
| `errored` | `emel::buffer::planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_plan_error>` | `idle` |
| `idle` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `resetting` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `seeding_leafs` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `counting_references` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `allocating_explicit_inputs` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `planning_nodes` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `releasing_expired` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `finalizing` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `splitting_required` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
| `errored` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::planner::action::on_unexpected>` | `errored` |
