# buffer_planner

Source: `emel/buffer_planner/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> resetting : emel::buffer_planner::event::plan [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::begin_plan>
  resetting --> seeding_leafs : emel::buffer_planner::event::reset_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_reset_done>
  resetting --> errored : emel::buffer_planner::event::reset_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::no_op>
  resetting --> errored : emel::buffer_planner::event::reset_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>
  seeding_leafs --> counting_references : emel::buffer_planner::event::seed_leafs_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_seed_leafs_done>
  seeding_leafs --> errored : emel::buffer_planner::event::seed_leafs_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_seed_leafs_done>
  seeding_leafs --> errored : emel::buffer_planner::event::seed_leafs_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>
  counting_references --> allocating_explicit_inputs : emel::buffer_planner::event::count_references_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_count_references_done>
  counting_references --> errored : emel::buffer_planner::event::count_references_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_count_references_done>
  counting_references --> errored : emel::buffer_planner::event::count_references_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>
  allocating_explicit_inputs --> planning_nodes : emel::buffer_planner::event::alloc_explicit_inputs_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_alloc_explicit_inputs_done>
  allocating_explicit_inputs --> errored : emel::buffer_planner::event::alloc_explicit_inputs_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_alloc_explicit_inputs_done>
  allocating_explicit_inputs --> errored : emel::buffer_planner::event::alloc_explicit_inputs_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>
  planning_nodes --> releasing_expired : emel::buffer_planner::event::plan_nodes_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_nodes_done>
  planning_nodes --> errored : emel::buffer_planner::event::plan_nodes_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_nodes_done>
  planning_nodes --> errored : emel::buffer_planner::event::plan_nodes_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>
  releasing_expired --> finalizing : emel::buffer_planner::event::release_expired_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_release_expired_done>
  releasing_expired --> errored : emel::buffer_planner::event::release_expired_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_release_expired_done>
  releasing_expired --> errored : emel::buffer_planner::event::release_expired_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>
  finalizing --> done : emel::buffer_planner::event::finalize_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_finalize_done>
  finalizing --> errored : emel::buffer_planner::event::finalize_done [boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_finalize_done>
  finalizing --> errored : emel::buffer_planner::event::finalize_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>
  resetting --> idle : emel::buffer_planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>
  seeding_leafs --> idle : emel::buffer_planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>
  counting_references --> idle : emel::buffer_planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>
  allocating_explicit_inputs --> idle : emel::buffer_planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>
  planning_nodes --> idle : emel::buffer_planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>
  releasing_expired --> idle : emel::buffer_planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>
  finalizing --> idle : emel::buffer_planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>
  errored --> idle : emel::buffer_planner::events::plan_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>
  done --> idle : emel::buffer_planner::events::plan_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_done>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `idle` | `emel::buffer_planner::event::plan` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::begin_plan>` | `resetting` |
| `resetting` | `emel::buffer_planner::event::reset_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_reset_done>` | `seeding_leafs` |
| `resetting` | `emel::buffer_planner::event::reset_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::no_op>` | `errored` |
| `resetting` | `emel::buffer_planner::event::reset_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>` | `errored` |
| `seeding_leafs` | `emel::buffer_planner::event::seed_leafs_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_seed_leafs_done>` | `counting_references` |
| `seeding_leafs` | `emel::buffer_planner::event::seed_leafs_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_seed_leafs_done>` | `errored` |
| `seeding_leafs` | `emel::buffer_planner::event::seed_leafs_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>` | `errored` |
| `counting_references` | `emel::buffer_planner::event::count_references_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_count_references_done>` | `allocating_explicit_inputs` |
| `counting_references` | `emel::buffer_planner::event::count_references_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_count_references_done>` | `errored` |
| `counting_references` | `emel::buffer_planner::event::count_references_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>` | `errored` |
| `allocating_explicit_inputs` | `emel::buffer_planner::event::alloc_explicit_inputs_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_alloc_explicit_inputs_done>` | `planning_nodes` |
| `allocating_explicit_inputs` | `emel::buffer_planner::event::alloc_explicit_inputs_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_alloc_explicit_inputs_done>` | `errored` |
| `allocating_explicit_inputs` | `emel::buffer_planner::event::alloc_explicit_inputs_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>` | `errored` |
| `planning_nodes` | `emel::buffer_planner::event::plan_nodes_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_nodes_done>` | `releasing_expired` |
| `planning_nodes` | `emel::buffer_planner::event::plan_nodes_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_nodes_done>` | `errored` |
| `planning_nodes` | `emel::buffer_planner::event::plan_nodes_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>` | `errored` |
| `releasing_expired` | `emel::buffer_planner::event::release_expired_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_release_expired_done>` | `finalizing` |
| `releasing_expired` | `emel::buffer_planner::event::release_expired_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_release_expired_done>` | `errored` |
| `releasing_expired` | `emel::buffer_planner::event::release_expired_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>` | `errored` |
| `finalizing` | `emel::buffer_planner::event::finalize_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_finalize_done>` | `done` |
| `finalizing` | `emel::buffer_planner::event::finalize_done` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_finalize_done>` | `errored` |
| `finalizing` | `emel::buffer_planner::event::finalize_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::record_phase_error>` | `errored` |
| `resetting` | `emel::buffer_planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>` | `idle` |
| `seeding_leafs` | `emel::buffer_planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>` | `idle` |
| `counting_references` | `emel::buffer_planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>` | `idle` |
| `allocating_explicit_inputs` | `emel::buffer_planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>` | `idle` |
| `planning_nodes` | `emel::buffer_planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>` | `idle` |
| `releasing_expired` | `emel::buffer_planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>` | `idle` |
| `finalizing` | `emel::buffer_planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>` | `idle` |
| `errored` | `emel::buffer_planner::events::plan_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_error>` | `idle` |
| `done` | `emel::buffer_planner::events::plan_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer_planner::action::on_plan_done>` | `idle` |
