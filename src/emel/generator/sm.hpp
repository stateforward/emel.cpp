#pragma once

#include <new>
#include <optional>
#include <utility>

#include "emel/generator/actions.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/guards.hpp"
#include "emel/generator/prefill/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/sm.hpp"
#include "emel/tensor/events.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"

namespace emel::generator {

namespace detail {

inline int32_t resolve_vocab_size(const emel::model::data & model) noexcept {
  const int32_t params_vocab = model.params.n_vocab;
  const int32_t model_vocab = static_cast<int32_t>(model.vocab_data.n_tokens);
  const int32_t use_params_vocab = static_cast<int32_t>(params_vocab > 0);
  return use_params_vocab * params_vocab + (1 - use_params_vocab) * model_vocab;
}

inline void reserve_session_buffers(action::context & ctx,
                                    const emel::model::data & model) noexcept {
  const int32_t vocab_size = resolve_vocab_size(model);
  const size_t allocation_size =
      static_cast<size_t>(vocab_size + static_cast<int32_t>(vocab_size <= 0));
  ctx.buffers.logits.reset(new (std::nothrow) float[allocation_size]);
  ctx.buffers.candidate_ids.reset(new (std::nothrow) int32_t[allocation_size]);
  ctx.buffers.candidate_scores.reset(new (std::nothrow) float[allocation_size]);
  ctx.buffers.candidate_capacity = vocab_size;
  ctx.buffers.vocab_size = vocab_size;
}

inline void dispatch_initialize_backend_error(const event::initialize & ev) noexcept {
  const auto err = emel::error::cast(error::backend);
  if (ev.error_out != nullptr) {
    *ev.error_out = err;
  }
  if (ev.on_error) {
    ev.on_error(events::initialize_error{&ev, err});
  }
}

inline bool dispatch_prefill_run(void * actor,
                                 const emel::generator::prefill::event::run & ev) noexcept {
  return static_cast<emel::generator::prefill::sm *>(actor)->process_event(ev);
}

}  // namespace detail

struct uninitialized {};
struct binding_conditioner {};
struct binding_conditioner_decision {};
struct initializing_renderer {};
struct initializing_renderer_decision {};
struct reserving_memory {};
struct reserving_memory_decision {};
struct reserving_graph {};
struct reserving_graph_decision {};
struct configuring_sampling_mode_decision {};
struct configuring_sampler {};
struct configuring_sampler_decision {};
struct configure_preselected_argmax {};
struct configure_preselected_argmax_decision {};
struct initialize_done_channel_decision {};
struct initialize_error_channel_decision {};

struct ready {};
struct reset_sequence {};
struct reset_sequence_decision {};
struct conditioning {};
struct conditioning_decision {};
struct planning {};
struct planning_decision {};
struct sequence_allocating {};
struct sequence_allocating_decision {};
struct prefill_running {};
struct prefill_result_decision {};
struct decode_slots {};
struct decode_slots_decision {};
struct snapshot_decode {};
struct snapshot_decode_decision {};
struct decode_compute_runtime_decision {};
struct decode_compute_flash {};
struct decode_compute_flash_preselected_argmax {};
struct decode_compute_flash_preselected_argmax_decision {};
struct decode_compute_flash_decision {};
struct decode_compute_nonflash {};
struct decode_compute_nonflash_preselected_argmax {};
struct decode_compute_nonflash_preselected_argmax_decision {};
struct decode_compute_nonflash_decision {};
struct decode_selection_mode_decision {};
struct decode_sample {};
struct decode_sample_decision {};
struct decode_preselected_argmax {};
struct decode_preselected_argmax_decision {};
struct decode_sample_preselected {};
struct decode_sample_preselected_decision {};
struct decode_render {};
struct decode_render_decision {};
struct decode_loop_decision {};
struct flushing {};
struct flushing_decision {};
struct generate_done_channel_decision {};
struct generate_ready_error_channel_decision {};
struct generate_uninitialized_error_channel_decision {};

/*
generator architecture notes (single source of truth)

state purpose
- initialize_* states bind injected dependencies and reserve owned session actors.
- ready is the only state that accepts generation.
- generate_* states orchestrate prompt conditioning, planning, memory reservation,
  graph execution, sampling, rendering, and final flush.

control invariants
- all runtime branching is modeled via explicit guards and decision states.
- request-scoped values live only in the typed runtime event ctx, not generator context.
- persistent session data lives only in generator context.
*/
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Session initialize validation.
        sml::state<binding_conditioner> <= *sml::state<uninitialized> + sml::event<event::initialize_run>
                 [ guard::valid_initialize{} ]
                 / action::begin_initialize

      , sml::state<initialize_error_channel_decision> <= sml::state<uninitialized>
                 + sml::event<event::initialize_run>
                 [ guard::invalid_initialize{} ]
                 / action::reject_initialize

      , sml::state<binding_conditioner> <= sml::state<ready> + sml::event<event::initialize_run>
                 [ guard::valid_initialize{} ]
                 / action::begin_initialize

      , sml::state<initialize_error_channel_decision> <= sml::state<ready>
                 + sml::event<event::initialize_run>
                 [ guard::invalid_initialize{} ]
                 / action::reject_initialize

      //------------------------------------------------------------------------------//
      // Initialize pipeline.
      , sml::state<binding_conditioner_decision> <= sml::state<binding_conditioner>
                 + sml::completion<event::initialize_run>
                 / action::request_conditioner_bind

      , sml::state<initializing_renderer> <= sml::state<binding_conditioner_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::conditioner_bind_ok{} ]

      , sml::state<initialize_error_channel_decision> <= sml::state<binding_conditioner_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::conditioner_bind_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<initialize_error_channel_decision> <= sml::state<binding_conditioner_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::conditioner_bind_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<initializing_renderer_decision> <= sml::state<initializing_renderer>
                 + sml::completion<event::initialize_run>
                 / action::request_renderer_initialize

      , sml::state<reserving_memory> <= sml::state<initializing_renderer_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::renderer_initialize_ok{} ]

      , sml::state<initialize_error_channel_decision> <= sml::state<initializing_renderer_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::renderer_initialize_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<initialize_error_channel_decision> <= sml::state<initializing_renderer_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::renderer_initialize_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<reserving_memory_decision> <= sml::state<reserving_memory>
                 + sml::completion<event::initialize_run>
                 / action::request_memory_reserve

      , sml::state<configuring_sampling_mode_decision> <= sml::state<reserving_memory_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::memory_reserve_with_existing_graph{} ]

      , sml::state<reserving_graph> <= sml::state<reserving_memory_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::memory_reserve_with_missing_graph{} ]

      , sml::state<initialize_error_channel_decision> <= sml::state<reserving_memory_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::memory_reserve_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<initialize_error_channel_decision> <= sml::state<reserving_memory_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::memory_reserve_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<reserving_graph_decision> <= sml::state<reserving_graph>
                 + sml::completion<event::initialize_run>
                 / action::request_graph_reserve

      , sml::state<configuring_sampling_mode_decision> <= sml::state<reserving_graph_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::graph_reserve_ok{} ]

      , sml::state<initialize_error_channel_decision> <= sml::state<reserving_graph_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::graph_reserve_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<initialize_error_channel_decision> <= sml::state<reserving_graph_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::graph_reserve_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<configuring_sampler> <= sml::state<configuring_sampling_mode_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_uses_materialized_logits{} ]

      , sml::state<configure_preselected_argmax> <= sml::state<configuring_sampling_mode_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_uses_preselected_argmax{} ]

      , sml::state<configuring_sampler_decision> <= sml::state<configuring_sampler>
                 + sml::completion<event::initialize_run>
                 / action::configure_sampler

      , sml::state<initialize_done_channel_decision> <= sml::state<configuring_sampler_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::sampler_configured{} ]

      , sml::state<initialize_error_channel_decision> <= sml::state<configuring_sampler_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::sampler_config_failed{} ]
                 / action::mark_backend_error

      , sml::state<configure_preselected_argmax_decision> <= sml::state<configure_preselected_argmax>
                 + sml::completion<event::initialize_run>
                 / action::configure_preselected_argmax

      , sml::state<initialize_done_channel_decision> <= sml::state<configure_preselected_argmax_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::sampler_configured{} ]

      , sml::state<initialize_error_channel_decision> <= sml::state<configure_preselected_argmax_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::sampler_config_failed{} ]
                 / action::mark_backend_error

      //------------------------------------------------------------------------------//
      // Initialize publication.
      , sml::state<ready> <= sml::state<initialize_done_channel_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_done_callback_with_error_out{} ]
                 / action::dispatch_initialize_done_with_callback_and_error_out

      , sml::state<ready> <= sml::state<initialize_done_channel_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_done_callback_without_error_out{} ]
                 / action::dispatch_initialize_done_with_callback_only

      , sml::state<ready> <= sml::state<initialize_done_channel_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_no_done_callback_with_error_out{} ]
                 / action::dispatch_initialize_done_with_error_out_only

      , sml::state<ready> <= sml::state<initialize_done_channel_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_no_done_callback_without_error_out{} ]
                 / action::dispatch_initialize_done_without_channels

      , sml::state<uninitialized> <= sml::state<initialize_error_channel_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_error_callback_with_error_out{} ]
                 / action::dispatch_initialize_error_with_callback_and_error_out

      , sml::state<uninitialized> <= sml::state<initialize_error_channel_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_error_callback_without_error_out{} ]
                 / action::dispatch_initialize_error_with_callback_only

      , sml::state<uninitialized> <= sml::state<initialize_error_channel_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_no_error_callback_with_error_out{} ]
                 / action::dispatch_initialize_error_with_error_out_only

      , sml::state<uninitialized> <= sml::state<initialize_error_channel_decision>
                 + sml::completion<event::initialize_run>
                 [ guard::initialize_no_error_callback_without_error_out{} ]
                 / action::dispatch_initialize_error_without_channels

      //------------------------------------------------------------------------------//
      // Generate validation.
      , sml::state<generate_uninitialized_error_channel_decision> <= sml::state<uninitialized>
                 + sml::event<event::generate_run>
                 / action::reject_uninitialized_generate

      , sml::state<reset_sequence> <= sml::state<ready> + sml::event<event::generate_run>
                 [ guard::valid_generate_with_reset{} ]
                 / action::begin_generate

      , sml::state<conditioning> <= sml::state<ready> + sml::event<event::generate_run>
                 [ guard::valid_generate_without_reset{} ]
                 / action::begin_generate

      , sml::state<generate_ready_error_channel_decision> <= sml::state<ready>
                 + sml::event<event::generate_run>
                 [ guard::invalid_generate{} ]
                 / action::reject_invalid_generate

      //------------------------------------------------------------------------------//
      // Session reset.
      , sml::state<reset_sequence_decision> <= sml::state<reset_sequence>
                 + sml::completion<event::generate_run>
                 / action::request_reset_sequence

      , sml::state<conditioning> <= sml::state<reset_sequence_decision>
                 + sml::completion<event::generate_run>
                 [ guard::reset_sequence_ok{} ]
                 / action::mark_sequence_clear

      , sml::state<generate_ready_error_channel_decision> <= sml::state<reset_sequence_decision>
                 + sml::completion<event::generate_run>
                 [ guard::reset_sequence_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<reset_sequence_decision>
                 + sml::completion<event::generate_run>
                 [ guard::reset_sequence_backend_error{} ]
                 / action::mark_backend_error

      //------------------------------------------------------------------------------//
      // Prompt conditioning.
      , sml::state<conditioning_decision> <= sml::state<conditioning>
                 + sml::completion<event::generate_run>
                 / action::request_conditioning

      , sml::state<planning> <= sml::state<conditioning_decision>
                 + sml::completion<event::generate_run>
                 [ guard::conditioning_ok{} ]

      , sml::state<generate_ready_error_channel_decision> <= sml::state<conditioning_decision>
                 + sml::completion<event::generate_run>
                 [ guard::conditioning_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<conditioning_decision>
                 + sml::completion<event::generate_run>
                 [ guard::conditioning_backend_error{} ]
                 / action::mark_backend_error

      //------------------------------------------------------------------------------//
      // Planning.
      , sml::state<planning_decision> <= sml::state<planning> + sml::completion<event::generate_run>
                 / action::request_planning

      , sml::state<sequence_allocating> <= sml::state<planning_decision>
                 + sml::completion<event::generate_run>
                 [ guard::planning_ok{} ]

      , sml::state<generate_ready_error_channel_decision> <= sml::state<planning_decision>
                 + sml::completion<event::generate_run>
                 [ guard::planning_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<planning_decision>
                 + sml::completion<event::generate_run>
                 [ guard::planning_backend_error{} ]
                 / action::mark_backend_error

      //------------------------------------------------------------------------------//
      // Sequence allocation.
      , sml::state<sequence_allocating_decision> <= sml::state<sequence_allocating>
                 + sml::completion<event::generate_run>
                 / action::request_allocate_sequence

      , sml::state<prefill_running> <= sml::state<sequence_allocating_decision>
                 + sml::completion<event::generate_run>
                 [ guard::allocate_sequence_ok{} ]
                 / action::mark_sequence_live

      , sml::state<generate_ready_error_channel_decision> <= sml::state<sequence_allocating_decision>
                 + sml::completion<event::generate_run>
                 [ guard::allocate_sequence_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<sequence_allocating_decision>
                 + sml::completion<event::generate_run>
                 [ guard::allocate_sequence_backend_error{} ]
                 / action::mark_backend_error

      //------------------------------------------------------------------------------//
      // Prefill.
      , sml::state<prefill_result_decision> <= sml::state<prefill_running>
                 + sml::completion<event::generate_run>
                 [ guard::prefill_dispatch_available{} ]
                 / action::request_prefill

      , sml::state<generate_ready_error_channel_decision> <= sml::state<prefill_running>
                 + sml::completion<event::generate_run>
                 [ guard::prefill_dispatch_unavailable{} ]
                 / action::mark_backend_error

      , sml::state<decode_selection_mode_decision> <= sml::state<prefill_result_decision>
                 + sml::completion<event::generate_run>
                 [ guard::prefill_result_ok_with_materialized_logits_contract{} ]

      , sml::state<decode_sample_preselected> <= sml::state<prefill_result_decision>
                 + sml::completion<event::generate_run>
                 [ guard::prefill_result_ok_with_preselected_argmax_contract{} ]

      , sml::state<generate_ready_error_channel_decision> <= sml::state<prefill_result_decision>
                 + sml::completion<event::generate_run>
                 [ guard::prefill_result_invalid_request{} ]

      , sml::state<generate_ready_error_channel_decision> <= sml::state<prefill_result_decision>
                 + sml::completion<event::generate_run>
                 [ guard::prefill_result_backend_error{} ]

      //------------------------------------------------------------------------------//
      // Decode loop.
      , sml::state<decode_sample> <= sml::state<decode_selection_mode_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_uses_materialized_logits{} ]

      , sml::state<decode_preselected_argmax> <= sml::state<decode_selection_mode_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_uses_preselected_argmax{} ]

      , sml::state<decode_sample_decision> <= sml::state<decode_sample>
                 + sml::completion<event::generate_run>
                 / action::request_decode_sample

      , sml::state<decode_preselected_argmax_decision> <= sml::state<decode_preselected_argmax>
                 + sml::completion<event::generate_run>
                 [ guard::decode_argmax_ready{} ]
                 / action::request_decode_select_argmax

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_preselected_argmax>
                 + sml::completion<event::generate_run>
                 [ guard::decode_argmax_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<decode_sample_preselected> <= sml::state<decode_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_ok{} ]

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<decode_sample_preselected_decision> <= sml::state<decode_sample_preselected>
                 + sml::completion<event::generate_run>
                 / action::request_decode_sample_preselected

      , sml::state<decode_render> <= sml::state<decode_sample_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_ok{} ]

      , sml::state<decode_render> <= sml::state<decode_sample_preselected_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_ok{} ]

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_sample_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_sample_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_sample_preselected_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_sample_preselected_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_sample_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<decode_render_decision> <= sml::state<decode_render>
                 + sml::completion<event::generate_run>
                 / action::request_decode_render

      , sml::state<decode_loop_decision> <= sml::state<decode_render_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_render_ok{} ]
                 / action::commit_render_output

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_render_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_render_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_render_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_render_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<decode_slots> <= sml::state<decode_loop_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_should_continue{} ]

      , sml::state<flushing> <= sml::state<decode_loop_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_complete{} ]

      , sml::state<decode_slots_decision> <= sml::state<decode_slots>
                 + sml::completion<event::generate_run>
                 / action::request_decode_slots

      , sml::state<snapshot_decode> <= sml::state<decode_slots_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_slots_ok{} ]

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_slots_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_slots_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_slots_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_slots_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<snapshot_decode_decision> <= sml::state<snapshot_decode>
                 + sml::completion<event::generate_run>
                 / action::request_memory_snapshot

      , sml::state<decode_compute_runtime_decision> <= sml::state<snapshot_decode_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_snapshot_ok{} ]

      , sml::state<generate_ready_error_channel_decision> <= sml::state<snapshot_decode_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_snapshot_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<snapshot_decode_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_snapshot_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<decode_compute_flash> <= sml::state<decode_compute_runtime_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_flash_runtime_supported{} ]

      , sml::state<decode_compute_nonflash> <= sml::state<decode_compute_runtime_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_nonflash_runtime_required{} ]

      , sml::state<decode_compute_flash_preselected_argmax> <= sml::state<decode_compute_flash>
                 + sml::completion<event::generate_run>
                 [ guard::compute_uses_preselected_argmax_direct{} ]

      , sml::state<decode_compute_flash_decision> <= sml::state<decode_compute_flash>
                 + sml::completion<event::generate_run>
                 [ guard::compute_uses_materialized_logits{} ]
                 / action::request_decode_compute_flash

      , sml::state<decode_compute_nonflash_preselected_argmax> <=
               sml::state<decode_compute_nonflash>
                 + sml::completion<event::generate_run>
                 [ guard::compute_uses_preselected_argmax_direct{} ]

      , sml::state<decode_compute_nonflash_decision> <= sml::state<decode_compute_nonflash>
                 + sml::completion<event::generate_run>
                 [ guard::compute_uses_materialized_logits{} ]
                 / action::request_decode_compute_nonflash

      , sml::state<decode_compute_flash_preselected_argmax_decision> <=
               sml::state<decode_compute_flash_preselected_argmax>
                 + sml::completion<event::generate_run>
                 / action::request_decode_compute_flash_preselected_argmax

      , sml::state<decode_compute_nonflash_preselected_argmax_decision> <=
               sml::state<decode_compute_nonflash_preselected_argmax>
                 + sml::completion<event::generate_run>
                 / action::request_decode_compute_nonflash_preselected_argmax

      , sml::state<decode_selection_mode_decision> <= sml::state<decode_compute_flash_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_ok{} ]
                 / action::advance_kv_cache

      , sml::state<decode_selection_mode_decision> <= sml::state<decode_compute_nonflash_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_ok{} ]
                 / action::advance_kv_cache

      , sml::state<decode_sample_preselected> <=
               sml::state<decode_compute_flash_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_ok{} ]
                 / action::advance_kv_cache

      , sml::state<decode_sample_preselected> <=
               sml::state<decode_compute_nonflash_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_ok{} ]
                 / action::advance_kv_cache

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_compute_flash_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_compute_flash_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_compute_nonflash_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<decode_compute_nonflash_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<generate_ready_error_channel_decision> <=
               sml::state<decode_compute_flash_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <=
               sml::state<decode_compute_flash_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<generate_ready_error_channel_decision> <=
               sml::state<decode_compute_nonflash_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <=
               sml::state<decode_compute_nonflash_preselected_argmax_decision>
                 + sml::completion<event::generate_run>
                 [ guard::decode_compute_backend_error{} ]
                 / action::mark_backend_error

      //------------------------------------------------------------------------------//
      // Final flush and publication.
      , sml::state<flushing_decision> <= sml::state<flushing>
                 + sml::completion<event::generate_run>
                 / action::request_flush

      , sml::state<generate_done_channel_decision> <= sml::state<flushing_decision>
                 + sml::completion<event::generate_run>
                 [ guard::flush_ok{} ]
                 / action::commit_flush_output

      , sml::state<generate_ready_error_channel_decision> <= sml::state<flushing_decision>
                 + sml::completion<event::generate_run>
                 [ guard::flush_invalid_request{} ]
                 / action::mark_invalid_request

      , sml::state<generate_ready_error_channel_decision> <= sml::state<flushing_decision>
                 + sml::completion<event::generate_run>
                 [ guard::flush_backend_error{} ]
                 / action::mark_backend_error

      , sml::state<ready> <= sml::state<generate_done_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_done_callback_with_error_out{} ]
                 / action::dispatch_generate_done_with_callback_and_error_out

      , sml::state<ready> <= sml::state<generate_done_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_done_callback_without_error_out{} ]
                 / action::dispatch_generate_done_with_callback_only

      , sml::state<ready> <= sml::state<generate_done_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_no_done_callback_with_error_out{} ]
                 / action::dispatch_generate_done_with_error_out_only

      , sml::state<ready> <= sml::state<generate_done_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_no_done_callback_without_error_out{} ]
                 / action::dispatch_generate_done_without_channels

      , sml::state<ready> <= sml::state<generate_ready_error_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_error_callback_with_error_out{} ]
                 / action::dispatch_generate_error_with_callback_and_error_out

      , sml::state<ready> <= sml::state<generate_ready_error_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_error_callback_without_error_out{} ]
                 / action::dispatch_generate_error_with_callback_only

      , sml::state<ready> <= sml::state<generate_ready_error_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_no_error_callback_with_error_out{} ]
                 / action::dispatch_generate_error_with_error_out_only

      , sml::state<ready> <= sml::state<generate_ready_error_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_no_error_callback_without_error_out{} ]
                 / action::dispatch_generate_error_without_channels

      , sml::state<uninitialized> <= sml::state<generate_uninitialized_error_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_error_callback_with_error_out{} ]
                 / action::dispatch_generate_error_with_callback_and_error_out

      , sml::state<uninitialized> <= sml::state<generate_uninitialized_error_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_error_callback_without_error_out{} ]
                 / action::dispatch_generate_error_with_callback_only

      , sml::state<uninitialized> <= sml::state<generate_uninitialized_error_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_no_error_callback_with_error_out{} ]
                 / action::dispatch_generate_error_with_error_out_only

      , sml::state<uninitialized> <= sml::state<generate_uninitialized_error_channel_decision>
                 + sml::completion<event::generate_run>
                 [ guard::generate_no_error_callback_without_error_out{} ]
                 / action::dispatch_generate_error_without_channels

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<uninitialized> <= sml::state<binding_conditioner> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<binding_conditioner_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<initializing_renderer> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<initializing_renderer_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<reserving_memory> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<reserving_memory_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<reserving_graph> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<reserving_graph_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<configuring_sampling_mode_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<configuring_sampler> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<configuring_sampler_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<configure_preselected_argmax> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<configure_preselected_argmax_decision>
                 + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<initialize_done_channel_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<initialize_error_channel_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<reset_sequence> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<reset_sequence_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<conditioning> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<conditioning_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<planning> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<planning_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<sequence_allocating> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<sequence_allocating_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<prefill_running> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<prefill_result_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_selection_mode_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_slots> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_slots_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<snapshot_decode> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<snapshot_decode_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_runtime_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_flash> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_flash_preselected_argmax>
                 + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_flash_preselected_argmax_decision>
                 + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_flash_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_nonflash> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_nonflash_preselected_argmax>
                 + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_nonflash_preselected_argmax_decision>
                 + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_compute_nonflash_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_sample> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_sample_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_preselected_argmax> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_preselected_argmax_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_sample_preselected> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_sample_preselected_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_render> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_render_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<decode_loop_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<flushing> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<flushing_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<generate_done_channel_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<generate_ready_error_channel_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<generate_uninitialized_error_channel_decision>
                 + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() : base_type() {
    prefill_actor_.emplace(emel::generator::prefill::action::context{this->context_});
    this->context_.prefill_actor = &prefill_actor_.value();
    this->context_.dispatch_prefill = detail::dispatch_prefill_run;
  }

  sm(const emel::model::data & model_ref,
     emel::text::conditioner::sm & conditioner_ref,
     void * formatter_ctx = nullptr,
     emel::text::formatter::format_fn format_prompt =
         emel::text::formatter::format_raw)
      : base_type() {
    prefill_actor_.emplace(emel::generator::prefill::action::context{this->context_});
    this->context_.prefill_actor = &prefill_actor_.value();
    this->context_.dispatch_prefill = detail::dispatch_prefill_run;
    this->context_.model = &model_ref;
    this->context_.conditioner = &conditioner_ref;
    this->context_.formatter_ctx = formatter_ctx;
    this->context_.format_prompt = format_prompt;
    // Session scratch is sized once from the injected loaded model so initialize stays allocation-free.
    detail::reserve_session_buffers(this->context_, model_ref);
    this->context_.compute.backend_ready =
        detail::prepare(this->context_.compute.backend, model_ref) ==
        emel::error::cast(emel::model::loader::error::none);
    if (this->context_.compute.backend_ready) {
      this->context_.compute.model_topology = this->context_.compute.backend.topology;
      this->context_.compute.prefill_plan = this->context_.compute.backend.prefill_plan;
      this->context_.compute.decode_plan = this->context_.compute.backend.decode_plan;
    }
  }

  sm(const sm &) = delete;
  sm(sm &&) = delete;
  sm & operator=(const sm &) = delete;
  sm & operator=(sm &&) = delete;

  bool process_event(const event::initialize & ev) {
    if (!this->context_.compute.backend_ready) {
      detail::dispatch_initialize_backend_error(ev);
      return false;
    }
    event::initialize_ctx ctx{};
    event::initialize_run runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::generate & ev) {
    event::generate_ctx ctx{};
    event::generate_run runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  emel::kernel::kernel_kind generation_kernel_kind() const noexcept {
    return this->context_.compute.backend.kernel_kind;
  }

  uint64_t generation_kernel_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel_dispatch_calls;
  }

  uint64_t generation_native_q8_0_dispatch_calls() const noexcept {
    return this->context_.compute.backend.native_q8_0_dispatch_calls;
  }

  uint64_t generation_packed_q8_0_dispatch_calls() const noexcept {
    return this->context_.compute.backend.packed_q8_0_dispatch_calls;
  }

  uint64_t generation_flash_attention_dispatch_calls() const noexcept {
    return this->context_.compute.backend.flash_attention_dispatch_calls;
  }

  uint64_t generation_optimized_flash_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.optimized_flash_dispatch_count();
  }

  uint64_t generation_shared_flash_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.shared_flash_dispatch_count();
  }

  uint64_t generation_optimized_q2_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.optimized_q2_dispatch_count();
  }

  uint64_t generation_shared_q2_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.shared_q2_dispatch_count();
  }

  uint64_t generation_optimized_q3_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.optimized_q3_dispatch_count();
  }

  uint64_t generation_shared_q3_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.shared_q3_dispatch_count();
  }

  uint64_t generation_optimized_q6_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.optimized_q6_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.optimized_q6_vector_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_argmax_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.optimized_q6_vector_argmax_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_packed_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.optimized_q6_vector_packed_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_packed_q8_rhs_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel
        .optimized_q6_vector_packed_q8_rhs_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_packed_q8_rhs_argmax_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel
        .optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_prepared_q8_rhs_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel
        .optimized_q6_vector_prepared_q8_rhs_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel
        .optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_calls()
      const noexcept {
    return this->context_.compute.backend.kernel
        .optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_count();
  }

  uint64_t generation_optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_calls()
      const noexcept {
    return this->context_.compute.backend.kernel
        .optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count();
  }

  uint64_t generation_shared_q6_dispatch_calls() const noexcept {
    return this->context_.compute.backend.kernel.shared_q6_dispatch_count();
  }

  uint32_t generation_native_quantized_stage_count() const noexcept {
    return detail::quantized_contract_stage_count(
        this->context_.compute.backend,
        emel::model::llama::detail::quantized_contract_kind::native_quantized);
  }

  uint32_t generation_approved_dense_f32_stage_count() const noexcept {
    return detail::quantized_contract_stage_count(
        this->context_.compute.backend,
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  }

  uint32_t generation_disallowed_fallback_stage_count() const noexcept {
    return detail::quantized_contract_stage_count(
        this->context_.compute.backend,
        emel::model::llama::detail::quantized_contract_kind::disallowed_fallback);
  }

  uint32_t generation_explicit_no_claim_stage_count() const noexcept {
    return detail::quantized_contract_stage_count(
        this->context_.compute.backend,
        emel::model::llama::detail::quantized_contract_kind::explicit_no_claim);
  }

  const emel::graph::event::reserve_output & graph_reservation() const noexcept {
    return this->context_.state.graph_reservation;
  }

  bool try_capture_graph_tensor(const int32_t tensor_id,
                                emel::tensor::event::tensor_state & state_out,
                                emel::error::type & err_out) noexcept {
    return this->context_.graph.try_capture_tensor(tensor_id, state_out, err_out);
  }

 private:
  std::optional<emel::generator::prefill::sm> prefill_actor_ = {};
};

}  // namespace emel::generator
