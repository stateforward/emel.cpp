#pragma once

// benchmark: designed

#include "emel/io/mmap/actions.hpp"
#include "emel/io/mmap/context.hpp"
#include "emel/io/mmap/detail.hpp"
#include "emel/io/mmap/events.hpp"
#include "emel/io/mmap/guards.hpp"
#include "emel/sm.hpp"

namespace emel::io::mmap {

struct state_ready {};
struct state_request_decision {};
struct state_file_path_decision {};
struct state_file_decision {};
struct state_offset_decision {};
struct state_length_decision {};
struct state_layout_decision {};
struct state_platform_decision {};
struct state_slot_reservation_decision {};
struct state_file_open_decision {};
struct state_file_size_decision {};
struct state_mapping_decision {};
struct state_done_callback {};
struct state_invalid_request_error_decision {};
struct state_unsupported_resource_error_decision {};
struct state_unsupported_platform_error_decision {};
struct state_resource_exhausted_error_decision {};
struct state_file_open_failed_error_decision {};
struct state_mapping_failed_error_decision {};
struct state_error_callback {};
struct state_release_decision {};
struct state_release_in_use_decision {};
struct state_unmap_decision {};
struct state_release_publish_done_decision {};
struct state_release_done_callback {};
struct state_release_invalid_handle_error_decision {};
struct state_unmap_failed_error_decision {};
struct state_release_error_callback {};
struct state_advise_decision {};
struct state_advise_owned_decision {};
struct state_advise_range_decision {};
struct state_advise_platform_decision {};
struct state_advise_kind_decision {};
struct state_advise_attempt_decision {};
struct state_advise_publish_done_decision {};
struct state_advise_done_callback {};
struct state_advise_invalid_handle_error_decision {};
struct state_advise_invalid_range_error_decision {};
struct state_advise_unsupported_platform_error_decision {};
struct state_advise_failed_error_decision {};
struct state_advise_invalid_kind_error_decision {};
struct state_advise_error_callback {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Map_tensor acceptance: every well-formed map_tensor enters the
      // validation chain. The chain explicitly walks request, file_path, file,
      // offset, length, layout, and platform preconditions before any platform
      // attempt is structurally reachable.
        sml::state<state_request_decision> <= *sml::state<state_ready>
          + sml::event<detail::map_tensor_runtime>
          / action::effect_begin_map_tensor

      //------------------------------------------------------------------------------//
      // Request validation.
      , sml::state<state_file_path_decision> <= sml::state<state_request_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::request_span_valid{} ]
      , sml::state<state_invalid_request_error_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::request_span_invalid{} ]
          / action::effect_mark_invalid_request

      //------------------------------------------------------------------------------//
      // File path validation.
      , sml::state<state_file_decision> <= sml::state<state_file_path_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::file_path_valid{} ]
      , sml::state<state_invalid_request_error_decision> <=
          sml::state<state_file_path_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::file_path_invalid{} ]
          / action::effect_mark_invalid_request

      //------------------------------------------------------------------------------//
      // File index validation.
      , sml::state<state_offset_decision> <= sml::state<state_file_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::file_index_valid{} ]
      , sml::state<state_unsupported_resource_error_decision> <=
          sml::state<state_file_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::file_index_invalid{} ]
          / action::effect_mark_unsupported_file

      //------------------------------------------------------------------------------//
      // Offset validation.
      , sml::state<state_length_decision> <= sml::state<state_offset_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::offset_aligned{} ]
      , sml::state<state_unsupported_resource_error_decision> <=
          sml::state<state_offset_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::offset_unaligned{} ]
          / action::effect_mark_unsupported_offset

      //------------------------------------------------------------------------------//
      // Length validation.
      , sml::state<state_layout_decision> <= sml::state<state_length_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::length_within_bounds{} ]
      , sml::state<state_unsupported_resource_error_decision> <=
          sml::state<state_length_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::length_overflow{} ]
          / action::effect_mark_unsupported_length

      //------------------------------------------------------------------------------//
      // Layout validation.
      , sml::state<state_platform_decision> <= sml::state<state_layout_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::layout_supported{} ]
      , sml::state<state_unsupported_resource_error_decision> <=
          sml::state<state_layout_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::layout_unsupported{} ]
          / action::effect_mark_unsupported_layout

      //------------------------------------------------------------------------------//
      // Platform validation. Phase 206 routes the supported branch into the
      // slot reservation chain that owns real OS attempts.
      , sml::state<state_slot_reservation_decision> <=
          sml::state<state_platform_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::platform_mmap_supported{} ]
      , sml::state<state_unsupported_platform_error_decision> <=
          sml::state<state_platform_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::platform_mmap_unsupported{} ]
          / action::effect_mark_unsupported_platform

      //------------------------------------------------------------------------------//
      // Slot reservation. Slot pool is fixed-capacity actor-owned state.
      , sml::state<state_file_open_decision> <=
          sml::state<state_slot_reservation_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::slot_capacity_available{} ]
          / action::effect_reserve_top_free_slot_then_attempt_open
      , sml::state<state_resource_exhausted_error_decision> <=
          sml::state<state_slot_reservation_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::slot_pool_exhausted{} ]
          / action::effect_mark_resource_exhausted

      //------------------------------------------------------------------------------//
      // File open decision. Entry action above already attempted the open and
      // recorded the raw result; this state routes success vs. failure.
      , sml::state<state_file_size_decision> <= sml::state<state_file_open_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::file_open_succeeded{} ]
          / action::effect_measure_open_file_size
      , sml::state<state_file_open_failed_error_decision> <=
          sml::state<state_file_open_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::file_open_failed{} ]
          / action::effect_release_reserved_slot_on_open_failure

      //------------------------------------------------------------------------------//
      // File size decision. The actor rejects spans beyond the opened file
      // before mapping, avoiding deferred SIGBUS-style failures on access.
      , sml::state<state_mapping_decision> <= sml::state<state_file_size_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::file_span_within_file{} ]
          / action::effect_attempt_mapping
      , sml::state<state_unsupported_resource_error_decision> <=
          sml::state<state_file_size_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::file_span_exceeds_file{} ]
          / action::effect_close_open_resource_and_release_slot_on_file_span_failure

      //------------------------------------------------------------------------------//
      // Mapping decision. Entry action above attempted mmap and recorded the
      // raw result; this state routes success vs. failure.
      , sml::state<state_done_callback> <= sml::state<state_mapping_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::mapping_succeeded{} ]
          / action::effect_commit_mapping
      , sml::state<state_mapping_failed_error_decision> <=
          sml::state<state_mapping_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::mapping_failed{} ]
          / action::effect_close_open_resource_and_release_slot_on_mapping_failure

      //------------------------------------------------------------------------------//
      // Done publication.
      , sml::state<state_ready> <= sml::state<state_done_callback>
          + sml::completion<detail::map_tensor_runtime>
          / action::effect_publish_map_tensor_done

      //------------------------------------------------------------------------------//
      // Map_tensor error publication for every error decision state.
      , sml::state<state_error_callback> <=
          sml::state<state_invalid_request_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_map_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_invalid_request_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_map_tensor_error
      , sml::state<state_error_callback> <=
          sml::state<state_unsupported_resource_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_map_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_unsupported_resource_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_map_tensor_error
      , sml::state<state_error_callback> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_map_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_map_tensor_error
      , sml::state<state_error_callback> <=
          sml::state<state_resource_exhausted_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_map_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_resource_exhausted_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_map_tensor_error
      , sml::state<state_error_callback> <=
          sml::state<state_file_open_failed_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_map_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_file_open_failed_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_map_tensor_error
      , sml::state<state_error_callback> <=
          sml::state<state_mapping_failed_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_map_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_mapping_failed_error_decision>
          + sml::completion<detail::map_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_map_tensor_error
      , sml::state<state_ready> <= sml::state<state_error_callback>
          + sml::completion<detail::map_tensor_runtime>
          / action::effect_record_map_tensor_error

      //------------------------------------------------------------------------------//
      // Release acceptance and validation chain.
      , sml::state<state_release_decision> <= sml::state<state_ready>
          + sml::event<detail::release_mapping_runtime>
          / action::effect_begin_release
      , sml::state<state_release_in_use_decision> <=
          sml::state<state_release_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_handle_in_range{} ]
      , sml::state<state_release_invalid_handle_error_decision> <=
          sml::state<state_release_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_handle_out_of_range{} ]
          / action::effect_mark_release_invalid_handle
      , sml::state<state_unmap_decision> <=
          sml::state<state_release_in_use_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_slot_in_use_owned_by_tensor{} ]
          / action::effect_attempt_unmap
      , sml::state<state_release_invalid_handle_error_decision> <=
          sml::state<state_release_in_use_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_slot_not_in_use{} ]
          / action::effect_mark_release_invalid_handle
      , sml::state<state_release_invalid_handle_error_decision> <=
          sml::state<state_release_in_use_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_slot_in_use_not_owned_by_tensor{} ]
          / action::effect_mark_release_invalid_handle
      , sml::state<state_release_publish_done_decision> <=
          sml::state<state_unmap_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::unmap_succeeded{} ]
          / action::effect_release_slot_after_unmap
      , sml::state<state_unmap_failed_error_decision> <=
          sml::state<state_unmap_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::unmap_failed{} ]
          / action::effect_mark_unmap_failed_and_release_slot

      //------------------------------------------------------------------------------//
      // Release done publication.
      , sml::state<state_release_done_callback> <=
          sml::state<state_release_publish_done_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_done_callback_present{} ]
          / action::effect_publish_release_mapping_done
      , sml::state<state_ready> <=
          sml::state<state_release_publish_done_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_done_callback_absent{} ]
          / action::effect_record_release_mapping_done
      , sml::state<state_ready> <= sml::state<state_release_done_callback>
          + sml::completion<detail::release_mapping_runtime>
          / action::effect_record_release_mapping_done

      //------------------------------------------------------------------------------//
      // Release error publication.
      , sml::state<state_release_error_callback> <=
          sml::state<state_release_invalid_handle_error_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_error_callback_present{} ]
          / action::effect_publish_release_mapping_error
      , sml::state<state_ready> <=
          sml::state<state_release_invalid_handle_error_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_error_callback_absent{} ]
          / action::effect_record_release_mapping_error
      , sml::state<state_release_error_callback> <=
          sml::state<state_unmap_failed_error_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_error_callback_present{} ]
          / action::effect_publish_release_mapping_error
      , sml::state<state_ready> <=
          sml::state<state_unmap_failed_error_decision>
          + sml::completion<detail::release_mapping_runtime>
          [ guard::release_error_callback_absent{} ]
          / action::effect_record_release_mapping_error
      , sml::state<state_ready> <= sml::state<state_release_error_callback>
          + sml::completion<detail::release_mapping_runtime>
          / action::effect_record_release_mapping_error

      //------------------------------------------------------------------------------//
      // Advise acceptance and validation chain: handle -> ownership -> range ->
      // platform, then explicit kind routing so each platform hint is its own
      // guarded row (no kind branching in effects).
      , sml::state<state_advise_decision> <= sml::state<state_ready>
          + sml::event<detail::advise_mapping_runtime>
          / action::effect_begin_advise
      , sml::state<state_advise_owned_decision> <= sml::state<state_advise_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_handle_in_range{} ]
      , sml::state<state_advise_invalid_handle_error_decision> <=
          sml::state<state_advise_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_handle_out_of_range{} ]
          / action::effect_mark_advise_invalid_handle
      , sml::state<state_advise_range_decision> <=
          sml::state<state_advise_owned_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_slot_in_use_owned_by_tensor{} ]
      , sml::state<state_advise_invalid_handle_error_decision> <=
          sml::state<state_advise_owned_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_slot_unavailable{} ]
          / action::effect_mark_advise_invalid_handle
      , sml::state<state_advise_platform_decision> <=
          sml::state<state_advise_range_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_range_within_mapping{} ]
      , sml::state<state_advise_invalid_range_error_decision> <=
          sml::state<state_advise_range_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_range_outside_mapping{} ]
          / action::effect_mark_advise_invalid_range
      , sml::state<state_advise_kind_decision> <=
          sml::state<state_advise_platform_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_platform_advise_supported{} ]
      , sml::state<state_advise_unsupported_platform_error_decision> <=
          sml::state<state_advise_platform_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_platform_advise_unsupported{} ]
          / action::effect_mark_advise_unsupported_platform
      , sml::state<state_advise_attempt_decision> <=
          sml::state<state_advise_kind_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_kind_sequential{} ]
          / action::effect_attempt_advise_sequential
      , sml::state<state_advise_attempt_decision> <=
          sml::state<state_advise_kind_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_kind_willneed{} ]
          / action::effect_attempt_advise_willneed
      , sml::state<state_advise_attempt_decision> <=
          sml::state<state_advise_kind_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_kind_dontneed{} ]
          / action::effect_attempt_advise_dontneed
      , sml::state<state_advise_invalid_kind_error_decision> <=
          sml::state<state_advise_kind_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_kind_invalid{} ]
          / action::effect_mark_advise_invalid_kind
      , sml::state<state_advise_publish_done_decision> <=
          sml::state<state_advise_attempt_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_succeeded{} ]
          / action::effect_commit_advise
      , sml::state<state_advise_failed_error_decision> <=
          sml::state<state_advise_attempt_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_failed{} ]
          / action::effect_mark_advise_failed

      //------------------------------------------------------------------------------//
      // Advise done publication.
      , sml::state<state_advise_done_callback> <=
          sml::state<state_advise_publish_done_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_done_callback_present{} ]
          / action::effect_publish_advise_mapping_done
      , sml::state<state_ready> <=
          sml::state<state_advise_publish_done_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_done_callback_absent{} ]
          / action::effect_record_advise_mapping_done
      , sml::state<state_ready> <= sml::state<state_advise_done_callback>
          + sml::completion<detail::advise_mapping_runtime>
          / action::effect_record_advise_mapping_done

      //------------------------------------------------------------------------------//
      // Advise error publication.
      , sml::state<state_advise_error_callback> <=
          sml::state<state_advise_invalid_handle_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_present{} ]
          / action::effect_publish_advise_mapping_error
      , sml::state<state_ready> <=
          sml::state<state_advise_invalid_handle_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_absent{} ]
          / action::effect_record_advise_mapping_error
      , sml::state<state_advise_error_callback> <=
          sml::state<state_advise_invalid_range_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_present{} ]
          / action::effect_publish_advise_mapping_error
      , sml::state<state_ready> <=
          sml::state<state_advise_invalid_range_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_absent{} ]
          / action::effect_record_advise_mapping_error
      , sml::state<state_advise_error_callback> <=
          sml::state<state_advise_unsupported_platform_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_present{} ]
          / action::effect_publish_advise_mapping_error
      , sml::state<state_ready> <=
          sml::state<state_advise_unsupported_platform_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_absent{} ]
          / action::effect_record_advise_mapping_error
      , sml::state<state_advise_error_callback> <=
          sml::state<state_advise_failed_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_present{} ]
          / action::effect_publish_advise_mapping_error
      , sml::state<state_ready> <=
          sml::state<state_advise_failed_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_absent{} ]
          / action::effect_record_advise_mapping_error
      , sml::state<state_advise_error_callback> <=
          sml::state<state_advise_invalid_kind_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_present{} ]
          / action::effect_publish_advise_mapping_error
      , sml::state<state_ready> <=
          sml::state<state_advise_invalid_kind_error_decision>
          + sml::completion<detail::advise_mapping_runtime>
          [ guard::guard_advise_error_callback_absent{} ]
          / action::effect_record_advise_mapping_error
      , sml::state<state_ready> <= sml::state<state_advise_error_callback>
          + sml::completion<detail::advise_mapping_runtime>
          / action::effect_record_advise_mapping_error

      //------------------------------------------------------------------------------//
      // Unexpected event handling for every reachable state.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_request_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_file_path_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_file_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_offset_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_length_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_layout_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_platform_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_slot_reservation_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_file_open_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_file_size_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_mapping_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_done_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_invalid_request_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_unsupported_resource_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_resource_exhausted_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_file_open_failed_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_mapping_failed_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_release_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_release_in_use_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_unmap_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_release_publish_done_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_release_done_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_release_invalid_handle_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_unmap_failed_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_release_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_advise_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_advise_owned_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_advise_range_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_advise_platform_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_advise_kind_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_advise_attempt_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_advise_publish_done_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_advise_done_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_advise_invalid_handle_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_advise_invalid_range_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_advise_invalid_kind_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_advise_unsupported_platform_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_advise_failed_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_advise_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::map_tensor &ev) {
    detail::map_attempt_status status{};
    detail::map_tensor_runtime runtime{ev, status};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }

  bool process_event(const event::release_mapping &ev) {
    detail::release_attempt_status status{};
    detail::release_mapping_runtime runtime{ev, status};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }

  bool process_event(const event::advise_mapping &ev) {
    detail::advise_attempt_status status{};
    detail::advise_mapping_runtime runtime{ev, status};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }
};

} // namespace emel::io::mmap
