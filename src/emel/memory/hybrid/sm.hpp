#pragma once
// benchmark: scaffold

#include <algorithm>
#include <array>
#include <memory>

#include "emel/memory/detail.hpp"
#include "emel/memory/hybrid/actions.hpp"
#include "emel/memory/hybrid/context.hpp"
#include "emel/memory/hybrid/events.hpp"
#include "emel/memory/hybrid/guards.hpp"
#include "emel/memory/view.hpp"
#include "emel/sm.hpp"

namespace emel::memory::hybrid {

struct ready {};

struct reserve_kv {};
struct reserve_kv_decision {};
struct reserve_recurrent {};
struct reserve_recurrent_decision {};

struct allocate_sequence_kv {};
struct allocate_sequence_kv_decision {};
struct allocate_sequence_recurrent {};
struct allocate_sequence_recurrent_decision {};
struct allocate_sequence_rollback_kv {};

struct allocate_slots_kv {};
struct allocate_slots_kv_decision {};
struct allocate_slots_recurrent {};
struct allocate_slots_recurrent_decision {};
struct allocate_slots_rollback_kv {};

struct branch_sequence_kv {};
struct branch_sequence_kv_decision {};
struct branch_sequence_recurrent {};
struct branch_sequence_recurrent_decision {};
struct branch_sequence_rollback_kv {};

struct free_sequence_kv {};
struct free_sequence_kv_decision {};
struct free_sequence_recurrent {};
struct free_sequence_recurrent_decision {};

struct rollback_slots_kv {};
struct rollback_slots_kv_decision {};
struct rollback_slots_recurrent {};
struct rollback_slots_recurrent_decision {};

struct capture_request_decision {};
struct capture_kv {};
struct capture_kv_decision {};
struct capture_recurrent {};
struct capture_recurrent_decision {};
struct capture_merge {};

struct done {};
struct out_of_memory {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<reserve_kv> <= *sml::state<ready> + sml::event<event::reserve_runtime>
          / action::begin_reserve
      , sml::state<reserve_kv_decision> <= sml::state<reserve_kv>
          + sml::completion<event::reserve_runtime> / action::exec_reserve_kv
      , sml::state<reserve_recurrent> <= sml::state<reserve_kv_decision>
          + sml::completion<event::reserve_runtime> [ guard::kv_accepted{} ]
      , sml::state<errored> <= sml::state<reserve_kv_decision>
          + sml::completion<event::reserve_runtime> [ guard::kv_rejected_with_error{} ]
          / action::mark_error_from_kv
      , sml::state<errored> <= sml::state<reserve_kv_decision>
          + sml::completion<event::reserve_runtime> [ guard::kv_rejected_without_error{} ]
          / action::mark_backend_error

      , sml::state<reserve_recurrent_decision> <= sml::state<reserve_recurrent>
          + sml::completion<event::reserve_runtime> / action::exec_reserve_recurrent
      , sml::state<done> <= sml::state<reserve_recurrent_decision>
          + sml::completion<event::reserve_runtime> [ guard::recurrent_accepted{} ]
      , sml::state<errored> <= sml::state<reserve_recurrent_decision>
          + sml::completion<event::reserve_runtime> [ guard::recurrent_rejected_with_error{} ]
          / action::mark_error_from_recurrent
      , sml::state<errored> <= sml::state<reserve_recurrent_decision>
          + sml::completion<event::reserve_runtime> [ guard::recurrent_rejected_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<allocate_sequence_kv> <= sml::state<ready>
          + sml::event<event::allocate_sequence_runtime> / action::begin_allocate_sequence
      , sml::state<allocate_sequence_kv_decision> <= sml::state<allocate_sequence_kv>
          + sml::completion<event::allocate_sequence_runtime> / action::exec_allocate_sequence_kv
      , sml::state<allocate_sequence_recurrent> <= sml::state<allocate_sequence_kv_decision>
          + sml::completion<event::allocate_sequence_runtime> [ guard::kv_accepted{} ]
      , sml::state<errored> <= sml::state<allocate_sequence_kv_decision>
          + sml::completion<event::allocate_sequence_runtime> [ guard::kv_rejected_with_error{} ]
          / action::mark_error_from_kv
      , sml::state<errored> <= sml::state<allocate_sequence_kv_decision>
          + sml::completion<event::allocate_sequence_runtime> [ guard::kv_rejected_without_error{} ]
          / action::mark_backend_error

      , sml::state<allocate_sequence_recurrent_decision> <= sml::state<allocate_sequence_recurrent>
          + sml::completion<event::allocate_sequence_runtime> / action::exec_allocate_sequence_recurrent
      , sml::state<done> <= sml::state<allocate_sequence_recurrent_decision>
          + sml::completion<event::allocate_sequence_runtime> [ guard::recurrent_accepted{} ]
      , sml::state<allocate_sequence_rollback_kv> <= sml::state<allocate_sequence_recurrent_decision>
          + sml::completion<event::allocate_sequence_runtime> [ guard::recurrent_rejected_any{} ]
          / action::exec_allocate_sequence_rollback_kv

      , sml::state<out_of_memory> <= sml::state<allocate_sequence_rollback_kv>
          + sml::completion<event::allocate_sequence_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_out_of_memory{} ]
          / action::mark_out_of_memory
      , sml::state<errored> <= sml::state<allocate_sequence_rollback_kv>
          + sml::completion<event::allocate_sequence_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_backend_or_none{} ]
          / action::mark_backend_error
      , sml::state<errored> <= sml::state<allocate_sequence_rollback_kv>
          + sml::completion<event::allocate_sequence_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_non_backend_error{} ]
          / action::mark_error_from_recurrent
      , sml::state<errored> <= sml::state<allocate_sequence_rollback_kv>
          + sml::completion<event::allocate_sequence_runtime> [ guard::rollback_rejected_with_error{} ]
          / action::mark_error_from_rollback
      , sml::state<errored> <= sml::state<allocate_sequence_rollback_kv>
          + sml::completion<event::allocate_sequence_runtime>
          [ guard::rollback_rejected_without_error{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<allocate_slots_kv> <= sml::state<ready>
          + sml::event<event::allocate_slots_runtime> / action::begin_allocate_slots
      , sml::state<allocate_slots_kv_decision> <= sml::state<allocate_slots_kv>
          + sml::completion<event::allocate_slots_runtime> / action::exec_allocate_slots_kv
      , sml::state<allocate_slots_recurrent> <= sml::state<allocate_slots_kv_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::kv_accepted{} ]
      , sml::state<out_of_memory> <= sml::state<allocate_slots_kv_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::kv_rejected_out_of_memory{} ]
          / action::mark_out_of_memory
      , sml::state<errored> <= sml::state<allocate_slots_kv_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::kv_rejected_backend_or_none{} ]
          / action::mark_backend_error
      , sml::state<errored> <= sml::state<allocate_slots_kv_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::kv_rejected_non_backend_error{} ]
          / action::mark_error_from_kv

      , sml::state<allocate_slots_recurrent_decision> <= sml::state<allocate_slots_recurrent>
          + sml::completion<event::allocate_slots_runtime> / action::exec_allocate_slots_recurrent
      , sml::state<done> <= sml::state<allocate_slots_recurrent_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::recurrent_accepted{} ]
      , sml::state<allocate_slots_rollback_kv> <= sml::state<allocate_slots_recurrent_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::recurrent_rejected_any{} ]
          / action::exec_allocate_slots_rollback_kv

      , sml::state<out_of_memory> <= sml::state<allocate_slots_rollback_kv>
          + sml::completion<event::allocate_slots_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_out_of_memory{} ]
          / action::mark_out_of_memory
      , sml::state<errored> <= sml::state<allocate_slots_rollback_kv>
          + sml::completion<event::allocate_slots_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_backend_or_none{} ]
          / action::mark_backend_error
      , sml::state<errored> <= sml::state<allocate_slots_rollback_kv>
          + sml::completion<event::allocate_slots_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_non_backend_error{} ]
          / action::mark_error_from_recurrent
      , sml::state<errored> <= sml::state<allocate_slots_rollback_kv>
          + sml::completion<event::allocate_slots_runtime> [ guard::rollback_rejected_with_error{} ]
          / action::mark_error_from_rollback
      , sml::state<errored> <= sml::state<allocate_slots_rollback_kv>
          + sml::completion<event::allocate_slots_runtime>
          [ guard::rollback_rejected_without_error{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<branch_sequence_kv> <= sml::state<ready>
          + sml::event<event::branch_sequence_runtime> / action::begin_branch_sequence
      , sml::state<branch_sequence_kv_decision> <= sml::state<branch_sequence_kv>
          + sml::completion<event::branch_sequence_runtime> / action::exec_branch_sequence_kv
      , sml::state<branch_sequence_recurrent> <= sml::state<branch_sequence_kv_decision>
          + sml::completion<event::branch_sequence_runtime> [ guard::kv_accepted{} ]
      , sml::state<out_of_memory> <= sml::state<branch_sequence_kv_decision>
          + sml::completion<event::branch_sequence_runtime> [ guard::kv_rejected_out_of_memory{} ]
          / action::mark_out_of_memory
      , sml::state<errored> <= sml::state<branch_sequence_kv_decision>
          + sml::completion<event::branch_sequence_runtime> [ guard::kv_rejected_backend_or_none{} ]
          / action::mark_backend_error
      , sml::state<errored> <= sml::state<branch_sequence_kv_decision>
          + sml::completion<event::branch_sequence_runtime> [ guard::kv_rejected_non_backend_error{} ]
          / action::mark_error_from_kv

      , sml::state<branch_sequence_recurrent_decision> <= sml::state<branch_sequence_recurrent>
          + sml::completion<event::branch_sequence_runtime> / action::exec_branch_sequence_recurrent
      , sml::state<done> <= sml::state<branch_sequence_recurrent_decision>
          + sml::completion<event::branch_sequence_runtime> [ guard::recurrent_accepted{} ]
      , sml::state<branch_sequence_rollback_kv> <= sml::state<branch_sequence_recurrent_decision>
          + sml::completion<event::branch_sequence_runtime> [ guard::recurrent_rejected_any{} ]
          / action::exec_branch_sequence_rollback_kv

      , sml::state<out_of_memory> <= sml::state<branch_sequence_rollback_kv>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_out_of_memory{} ]
          / action::mark_out_of_memory
      , sml::state<errored> <= sml::state<branch_sequence_rollback_kv>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_backend_or_none{} ]
          / action::mark_backend_error
      , sml::state<errored> <= sml::state<branch_sequence_rollback_kv>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::rollback_accepted_and_recurrent_rejected_non_backend_error{} ]
          / action::mark_error_from_recurrent
      , sml::state<errored> <= sml::state<branch_sequence_rollback_kv>
          + sml::completion<event::branch_sequence_runtime> [ guard::rollback_rejected_with_error{} ]
          / action::mark_error_from_rollback
      , sml::state<errored> <= sml::state<branch_sequence_rollback_kv>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::rollback_rejected_without_error{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<free_sequence_kv> <= sml::state<ready>
          + sml::event<event::free_sequence_runtime> / action::begin_free_sequence
      , sml::state<free_sequence_kv_decision> <= sml::state<free_sequence_kv>
          + sml::completion<event::free_sequence_runtime> / action::exec_free_sequence_kv
      , sml::state<free_sequence_recurrent> <= sml::state<free_sequence_kv_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::kv_accepted{} ]
      , sml::state<errored> <= sml::state<free_sequence_kv_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::kv_rejected_with_error{} ]
          / action::mark_error_from_kv
      , sml::state<errored> <= sml::state<free_sequence_kv_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::kv_rejected_without_error{} ]
          / action::mark_backend_error

      , sml::state<free_sequence_recurrent_decision> <= sml::state<free_sequence_recurrent>
          + sml::completion<event::free_sequence_runtime> / action::exec_free_sequence_recurrent
      , sml::state<done> <= sml::state<free_sequence_recurrent_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::recurrent_accepted{} ]
      , sml::state<errored> <= sml::state<free_sequence_recurrent_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::recurrent_rejected_with_error{} ]
          / action::mark_error_from_recurrent
      , sml::state<errored> <= sml::state<free_sequence_recurrent_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::recurrent_rejected_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<rollback_slots_kv> <= sml::state<ready>
          + sml::event<event::rollback_slots_runtime> / action::begin_rollback_slots
      , sml::state<rollback_slots_kv_decision> <= sml::state<rollback_slots_kv>
          + sml::completion<event::rollback_slots_runtime> / action::exec_rollback_slots_kv
      , sml::state<rollback_slots_recurrent> <= sml::state<rollback_slots_kv_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::kv_accepted{} ]
      , sml::state<errored> <= sml::state<rollback_slots_kv_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::kv_rejected_with_error{} ]
          / action::mark_error_from_kv
      , sml::state<errored> <= sml::state<rollback_slots_kv_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::kv_rejected_without_error{} ]
          / action::mark_backend_error

      , sml::state<rollback_slots_recurrent_decision> <= sml::state<rollback_slots_recurrent>
          + sml::completion<event::rollback_slots_runtime> / action::exec_rollback_slots_recurrent
      , sml::state<done> <= sml::state<rollback_slots_recurrent_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::recurrent_accepted{} ]
      , sml::state<errored> <= sml::state<rollback_slots_recurrent_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::recurrent_rejected_with_error{} ]
          / action::mark_error_from_recurrent
      , sml::state<errored> <= sml::state<rollback_slots_recurrent_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::recurrent_rejected_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<capture_request_decision> <= sml::state<ready>
          + sml::event<event::capture_view_runtime> / action::begin_capture_view
      , sml::state<capture_kv> <= sml::state<capture_request_decision>
          + sml::completion<event::capture_view_runtime> [ guard::capture_request_valid{} ]
      , sml::state<errored> <= sml::state<capture_request_decision>
          + sml::completion<event::capture_view_runtime> [ guard::capture_request_invalid{} ]
          / action::mark_invalid_request

      , sml::state<capture_kv_decision> <= sml::state<capture_kv>
          + sml::completion<event::capture_view_runtime> / action::exec_capture_kv
      , sml::state<capture_recurrent> <= sml::state<capture_kv_decision>
          + sml::completion<event::capture_view_runtime> [ guard::kv_accepted{} ]
      , sml::state<errored> <= sml::state<capture_kv_decision>
          + sml::completion<event::capture_view_runtime> [ guard::kv_rejected_with_error{} ]
          / action::mark_error_from_kv
      , sml::state<errored> <= sml::state<capture_kv_decision>
          + sml::completion<event::capture_view_runtime> [ guard::kv_rejected_without_error{} ]
          / action::mark_backend_error

      , sml::state<capture_recurrent_decision> <= sml::state<capture_recurrent>
          + sml::completion<event::capture_view_runtime> / action::exec_capture_recurrent
      , sml::state<capture_merge> <= sml::state<capture_recurrent_decision>
          + sml::completion<event::capture_view_runtime> [ guard::recurrent_accepted{} ]
      , sml::state<errored> <= sml::state<capture_recurrent_decision>
          + sml::completion<event::capture_view_runtime> [ guard::recurrent_rejected_with_error{} ]
          / action::mark_error_from_recurrent
      , sml::state<errored> <= sml::state<capture_recurrent_decision>
          + sml::completion<event::capture_view_runtime> [ guard::recurrent_rejected_without_error{} ]
          / action::mark_backend_error

      , sml::state<done> <= sml::state<capture_merge>
          + sml::completion<event::capture_view_runtime> / action::merge_capture_snapshots

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<event::reserve_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<out_of_memory> + sml::completion<event::reserve_runtime>
          / action::publish_error
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::reserve_runtime>
          / action::publish_error

      , sml::state<ready> <= sml::state<done> + sml::completion<event::allocate_sequence_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<out_of_memory>
          + sml::completion<event::allocate_sequence_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::allocate_sequence_runtime> / action::publish_error

      , sml::state<ready> <= sml::state<done> + sml::completion<event::allocate_slots_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<out_of_memory>
          + sml::completion<event::allocate_slots_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::allocate_slots_runtime> / action::publish_error

      , sml::state<ready> <= sml::state<done> + sml::completion<event::branch_sequence_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<out_of_memory>
          + sml::completion<event::branch_sequence_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::branch_sequence_runtime> / action::publish_error

      , sml::state<ready> <= sml::state<done> + sml::completion<event::free_sequence_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<out_of_memory>
          + sml::completion<event::free_sequence_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::free_sequence_runtime> / action::publish_error

      , sml::state<ready> <= sml::state<done> + sml::completion<event::rollback_slots_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<out_of_memory>
          + sml::completion<event::rollback_slots_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::rollback_slots_runtime> / action::publish_error

      , sml::state<ready> <= sml::state<done> + sml::completion<event::capture_view_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<out_of_memory>
          + sml::completion<event::capture_view_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::capture_view_runtime> / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_kv> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_kv_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_recurrent> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_recurrent_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_sequence_kv> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_sequence_kv_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_sequence_recurrent>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_sequence_recurrent_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_sequence_rollback_kv>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_kv> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_kv_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_recurrent> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_recurrent_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_rollback_kv>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_kv> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_kv_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_recurrent> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_recurrent_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_rollback_kv>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<free_sequence_kv> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<free_sequence_kv_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<free_sequence_recurrent> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<free_sequence_recurrent_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<rollback_slots_kv> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<rollback_slots_kv_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<rollback_slots_recurrent> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<rollback_slots_recurrent_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_kv> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_kv_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_recurrent> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_recurrent_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_merge> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<out_of_memory> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<errored> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  sm()
      : base_type(),
        kv_snapshot_(std::make_unique<view::snapshot>()),
        recurrent_snapshot_(std::make_unique<view::snapshot>()),
        snapshot_(std::make_unique<view::snapshot>()) {}

  bool process_event(const event::reserve & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::reserve_ctx ctx{};
    event::reserve_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::allocate_sequence & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::allocate_sequence_ctx ctx{};
    event::allocate_sequence_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::allocate_slots & ev) {
    int32_t block_count_sink = 0;
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::allocate_slots_ctx ctx{};
    event::allocate_slots_runtime runtime{
        ev, ctx,
        emel::memory::detail::bind_or_sink(ev.block_count_out, block_count_sink),
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::branch_sequence & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::branch_sequence_ctx ctx{};
    event::branch_sequence_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::free_sequence & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::free_sequence_ctx ctx{};
    event::free_sequence_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::rollback_slots & ev) {
    int32_t block_count_sink = 0;
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::rollback_slots_ctx ctx{};
    event::rollback_slots_runtime runtime{
        ev, ctx,
        emel::memory::detail::bind_or_sink(ev.block_count_out, block_count_sink),
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::capture_view & ev) {
    view::snapshot & snapshot_out =
        emel::memory::detail::bind_or_sink(ev.snapshot_out, *snapshot_);
    const bool writes_cached_snapshot = &snapshot_out == snapshot_.get();
    const bool has_snapshot_out = ev.snapshot_out != nullptr;
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::capture_view_ctx ctx{};
    event::capture_view_runtime runtime{
        ev,
        ctx,
        *kv_snapshot_,
        *recurrent_snapshot_,
        snapshot_out,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink),
        has_snapshot_out};
    const bool accepted = base_type::process_event(runtime);
    const bool ok = accepted && ctx.err == emel::error::cast(error::none);
    snapshot_dirty_ = snapshot_dirty_ && !(ok && writes_cached_snapshot);
    return ok;
  }

  bool try_view(view::snapshot & snapshot_out, emel::error::type & err_out) noexcept {
    int32_t err = static_cast<int32_t>(emel::error::cast(error::none));
    const bool accepted = process_event(event::capture_view{
      .snapshot_out = &snapshot_out,
      .error_out = &err,
    });
    err_out = static_cast<emel::error::type>(err);
    return accepted && err_out == emel::error::cast(error::none);
  }

  const view::snapshot & view() noexcept {
    const int32_t was_dirty = static_cast<int32_t>(snapshot_dirty_);
    emel::error::type err = emel::error::cast(error::none);
    const std::array<bool (*)(sm &, view::snapshot &, emel::error::type &) noexcept, 2> refreshers{
        &sm::refresh_noop,
        &sm::refresh_try_view,
    };
    const bool refreshed =
        refreshers[static_cast<size_t>(snapshot_dirty_)](*this, *snapshot_, err);
    const int32_t refreshed_int = static_cast<int32_t>(refreshed);
    const int32_t refresh_failed = was_dirty * (1 - refreshed_int);
    const std::array<void (*)(view::snapshot &) noexcept, 2> clearers{
        &emel::memory::detail::clear_snapshot_noop,
        &emel::memory::detail::clear_snapshot,
    };
    clearers[static_cast<size_t>(refresh_failed)](*snapshot_);
    snapshot_dirty_ = static_cast<bool>(refresh_failed);
    return *snapshot_;
  }

 private:
  static bool refresh_noop(sm &, view::snapshot &, emel::error::type &) noexcept { return true; }

  static bool refresh_try_view(sm & self, view::snapshot & snapshot_out,
                               emel::error::type & err_out) noexcept {
    return self.try_view(snapshot_out, err_out);
  }

  std::unique_ptr<view::snapshot> kv_snapshot_;
  std::unique_ptr<view::snapshot> recurrent_snapshot_;
  std::unique_ptr<view::snapshot> snapshot_;
  bool snapshot_dirty_ = true;
};

using Hybrid = sm;

}  // namespace emel::memory::hybrid
