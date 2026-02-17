#pragma once

#include "emel/buffer/chunk_allocator/actions.hpp"
#include "emel/buffer/chunk_allocator/events.hpp"
#include "emel/buffer/chunk_allocator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::buffer::chunk_allocator {

using Process = boost::sml::back::process<
  event::validate_configure,
  events::validate_configure_done,
  events::validate_configure_error,
  event::apply_configure,
  events::apply_configure_done,
  events::apply_configure_error,
  events::configure_done,
  events::configure_error,
  event::validate_allocate,
  events::validate_allocate_done,
  events::validate_allocate_error,
  event::select_block,
  events::select_block_done,
  events::select_block_error,
  event::ensure_chunk,
  events::ensure_chunk_done,
  events::ensure_chunk_error,
  event::commit_allocate,
  events::commit_allocate_done,
  events::commit_allocate_error,
  events::allocate_done,
  events::allocate_error,
  event::validate_release,
  events::validate_release_done,
  events::validate_release_error,
  event::merge_release,
  events::merge_release_done,
  events::merge_release_error,
  events::release_done,
  events::release_error,
  event::apply_reset,
  events::apply_reset_done,
  events::apply_reset_error,
  events::reset_done,
  events::reset_error>;

/**
 * Dynamic chunk allocator orchestration model.
 *
 * Parity reference:
 * - `ggml_dyn_tallocr_alloc(...)`
 * - `ggml_dyn_tallocr_free_bytes(...)`
 * - `ggml_dyn_tallocr_reset(...)`
 *
 * State purposes:
 * - `ready`: accepts configure, allocate, release, and reset operations.
 * - `configuring`/`applying_configure`: validates and applies runtime allocator limits.
 * - `validating_allocate`/`selecting_block`/`ensuring_chunk`/`committing_allocate`:
 *   allocation pipeline with best-fit and chunk growth behavior.
 * - `validating_release`/`merging_release`: free-bytes merge pipeline.
 * - `resetting`: clears chunk/free-block runtime state.
 * - `failed`: operation failure terminal before operation-specific `_error`.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    struct ready {};
    struct configuring {};
    struct applying_configure {};
    struct configure_done {};
    struct validating_allocate {};
    struct selecting_block {};
    struct ensuring_chunk {};
    struct committing_allocate {};
    struct allocate_done {};
    struct validating_release {};
    struct merging_release {};
    struct release_done {};
    struct resetting {};
    struct reset_done {};
    struct failed {};

    return sml::make_transition_table(
      *sml::state<ready> + sml::event<event::configure> / action::begin_configure =
          sml::state<configuring>,
      sml::state<configuring> + sml::on_entry<event::configure> /
          [](const event::configure & ev, action::context &, process_t & process) noexcept {
            process(event::validate_configure{
              .error_out = ev.error_out,
              .request = &ev,
            });
          },
      sml::state<configuring> + sml::event<event::validate_configure> [guard::valid_configure{}] /
          [](const event::validate_configure & ev, action::context & ctx,
             process_t & process) noexcept {
            if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
            ctx.step += 1;
            process(events::validate_configure_done{.request = ev.request});
          } = sml::state<configuring>,
      sml::state<configuring> + sml::event<event::validate_configure> [guard::invalid_configure{}] /
          [](const event::validate_configure & ev, action::context & ctx,
             process_t & process) noexcept {
            if (ev.error_out != nullptr) *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
            ctx.step += 1;
            process(events::validate_configure_error{
              .err = EMEL_ERR_INVALID_ARGUMENT,
              .request = ev.request,
            });
          } = sml::state<configuring>,
      sml::state<configuring> + sml::event<events::validate_configure_done> =
          sml::state<applying_configure>,
      sml::state<configuring> + sml::event<events::validate_configure_error> = sml::state<failed>,
      sml::state<applying_configure> + sml::on_entry<events::validate_configure_done> /
          [](const events::validate_configure_done & ev, action::context &,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::apply_configure apply{
              .error_out = &phase_error,
            };
            process(apply);
            if (phase_error != EMEL_OK) {
              process(events::apply_configure_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::apply_configure_done{
              .request = ev.request,
            });
          },
      sml::state<applying_configure> + sml::event<event::apply_configure> /
          action::run_apply_configure = sml::state<applying_configure>,
      sml::state<applying_configure> + sml::event<events::apply_configure_done> =
          sml::state<configure_done>,
      sml::state<applying_configure> + sml::event<events::apply_configure_error> =
          sml::state<failed>,
      sml::state<configure_done> + sml::on_entry<events::apply_configure_done> /
          [](const events::apply_configure_done & ev, action::context &,
             process_t & process) noexcept {
            const event::configure * request = ev.request;
            process(events::configure_done{
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<configure_done> + sml::event<events::configure_done> / action::on_configure_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::allocate> / action::begin_allocate =
          sml::state<validating_allocate>,
      sml::state<validating_allocate> + sml::on_entry<event::allocate> /
          [](const event::allocate & ev, action::context &, process_t & process) noexcept {
            process(event::validate_allocate{
              .error_out = ev.error_out,
              .request = &ev,
            });
          },
      sml::state<validating_allocate> + sml::event<event::validate_allocate>
          [guard::valid_allocate_request{}] /
          [](const event::validate_allocate & ev, action::context & ctx,
             process_t & process) noexcept {
            uint64_t aligned = 0;
            action::detail::align_up(ctx.request_size, ctx.request_alignment, aligned);
            ctx.aligned_request_size = aligned;
            if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
            ctx.step += 1;
            process(events::validate_allocate_done{.request = ev.request});
          } =
          sml::state<validating_allocate>,
      sml::state<validating_allocate> + sml::event<event::validate_allocate>
          [guard::invalid_allocate_request{}] /
          [](const event::validate_allocate & ev, action::context & ctx,
             process_t & process) noexcept {
            if (ev.error_out != nullptr) *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
            ctx.step += 1;
            process(events::validate_allocate_error{
              .err = EMEL_ERR_INVALID_ARGUMENT,
              .request = ev.request,
            });
          } =
          sml::state<validating_allocate>,
      sml::state<validating_allocate> + sml::event<events::validate_allocate_done> =
          sml::state<selecting_block>,
      sml::state<validating_allocate> + sml::event<events::validate_allocate_error> =
          sml::state<failed>,
      sml::state<selecting_block> + sml::on_entry<events::validate_allocate_done> /
          [](const events::validate_allocate_done & ev, action::context &,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::select_block select{
              .error_out = &phase_error,
            };
            process(select);
            if (phase_error != EMEL_OK) {
              process(events::select_block_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::select_block_done{
              .request = ev.request,
            });
          },
      sml::state<selecting_block> + sml::event<event::select_block> / action::run_select_block =
          sml::state<selecting_block>,
      sml::state<selecting_block> + sml::event<events::select_block_done> =
          sml::state<ensuring_chunk>,
      sml::state<selecting_block> + sml::event<events::select_block_error> = sml::state<failed>,
      sml::state<ensuring_chunk> + sml::on_entry<events::select_block_done> /
          [](const events::select_block_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::ensure_chunk ensure{
              .error_out = &phase_error,
            };
            process(ensure);
            if (phase_error != EMEL_OK) {
              process(events::ensure_chunk_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::ensure_chunk_done{
              .request = ev.request,
            });
          },
      sml::state<ensuring_chunk> + sml::event<event::ensure_chunk> / action::run_ensure_chunk =
          sml::state<ensuring_chunk>,
      sml::state<ensuring_chunk> + sml::event<events::ensure_chunk_done> =
          sml::state<committing_allocate>,
      sml::state<ensuring_chunk> + sml::event<events::ensure_chunk_error> = sml::state<failed>,
      sml::state<committing_allocate> + sml::on_entry<events::ensure_chunk_done> /
          [](const events::ensure_chunk_done & ev, action::context &,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::commit_allocate commit{
              .error_out = &phase_error,
            };
            process(commit);
            if (phase_error != EMEL_OK) {
              process(events::commit_allocate_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::commit_allocate_done{
              .request = ev.request,
            });
          },
      sml::state<committing_allocate> + sml::event<event::commit_allocate> /
          action::run_commit_allocate = sml::state<committing_allocate>,
      sml::state<committing_allocate> + sml::event<events::commit_allocate_done> =
          sml::state<allocate_done>,
      sml::state<committing_allocate> + sml::event<events::commit_allocate_error> =
          sml::state<failed>,
      sml::state<allocate_done> + sml::on_entry<events::commit_allocate_done> /
          [](const events::commit_allocate_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::allocate * request = ev.request;
            if (request != nullptr) {
              if (request->chunk_out != nullptr) {
                *request->chunk_out = ctx.result_chunk;
              }
              if (request->offset_out != nullptr) {
                *request->offset_out = ctx.result_offset;
              }
              if (request->aligned_size_out != nullptr) {
                *request->aligned_size_out = ctx.result_size;
              }
            }
            process(events::allocate_done{
              .chunk = ctx.result_chunk,
              .offset = ctx.result_offset,
              .size = ctx.result_size,
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<allocate_done> + sml::event<events::allocate_done> / action::on_allocate_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::release> / action::begin_release =
          sml::state<validating_release>,
      sml::state<validating_release> + sml::on_entry<event::release> /
          [](const event::release & ev, action::context &, process_t & process) noexcept {
            process(event::validate_release{
              .error_out = ev.error_out,
              .request = &ev,
            });
          },
      sml::state<validating_release> + sml::event<event::validate_release>
          [guard::valid_release_request{}] /
          [](const event::validate_release & ev, action::context & ctx,
             process_t & process) noexcept {
            uint64_t aligned = 0;
            action::detail::align_up(ctx.request_size, ctx.request_alignment, aligned);
            ctx.aligned_request_size = aligned;
            if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
            ctx.step += 1;
            process(events::validate_release_done{.request = ev.request});
          } =
          sml::state<validating_release>,
      sml::state<validating_release> + sml::event<event::validate_release>
          [guard::invalid_release_request{}] /
          [](const event::validate_release & ev, action::context & ctx,
             process_t & process) noexcept {
            if (ev.error_out != nullptr) *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
            ctx.step += 1;
            process(events::validate_release_error{
              .err = EMEL_ERR_INVALID_ARGUMENT,
              .request = ev.request,
            });
          } =
          sml::state<validating_release>,
      sml::state<validating_release> + sml::event<events::validate_release_done> =
          sml::state<merging_release>,
      sml::state<validating_release> + sml::event<events::validate_release_error> =
          sml::state<failed>,
      sml::state<merging_release> + sml::on_entry<events::validate_release_done> /
          [](const events::validate_release_done & ev, action::context &,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::merge_release merge{
              .error_out = &phase_error,
            };
            process(merge);
            if (phase_error != EMEL_OK) {
              process(events::merge_release_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::merge_release_done{
              .request = ev.request,
            });
          },
      sml::state<merging_release> + sml::event<event::merge_release> / action::run_merge_release =
          sml::state<merging_release>,
      sml::state<merging_release> + sml::event<events::merge_release_done> =
          sml::state<release_done>,
      sml::state<merging_release> + sml::event<events::merge_release_error> = sml::state<failed>,
      sml::state<release_done> + sml::on_entry<events::merge_release_done> /
          [](const events::merge_release_done & ev, action::context &, process_t & process) noexcept {
            const event::release * request = ev.request;
            process(events::release_done{
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<release_done> + sml::event<events::release_done> / action::on_release_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<configuring> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<applying_configure> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<configure_done> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<validating_allocate> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<selecting_block> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<ensuring_chunk> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<committing_allocate> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<allocate_done> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<validating_release> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<merging_release> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<release_done> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<failed> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<resetting> + sml::event<event::apply_reset> / action::run_apply_reset =
          sml::state<resetting>,
      sml::state<resetting> + sml::event<events::apply_reset_done> = sml::state<reset_done>,
      sml::state<resetting> + sml::event<events::apply_reset_error> = sml::state<failed>,
      sml::state<resetting> + sml::on_entry<event::reset> /
          [](const event::reset & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::apply_reset apply{
              .error_out = &phase_error,
            };
            process(apply);
            if (phase_error != EMEL_OK) {
              process(events::apply_reset_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::apply_reset_done{
              .request = &ev,
            });
          },
      sml::state<reset_done> + sml::on_entry<events::apply_reset_done> /
          [](const events::apply_reset_done & ev, action::context &, process_t & process) noexcept {
            const event::reset * request = ev.request;
            process(events::reset_done{
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<reset_done> + sml::event<events::reset_done> / action::on_reset_done =
          sml::state<ready>,

      sml::state<failed> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_INVALID_ARGUMENT;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              if constexpr (std::is_same_v<decltype(ev.request), const event::configure *>) {
                const event::configure * request = ev.request;
                process(events::configure_error{
                  .err = err,
                  .error_out = request != nullptr ? request->error_out : nullptr,
                  .request = request,
                });
                return;
              } else if constexpr (std::is_same_v<decltype(ev.request), const event::allocate *>) {
                const event::allocate * request = ev.request;
                process(events::allocate_error{
                  .err = err,
                  .error_out = request != nullptr ? request->error_out : nullptr,
                  .request = request,
                });
                return;
              } else if constexpr (std::is_same_v<decltype(ev.request), const event::release *>) {
                const event::release * request = ev.request;
                process(events::release_error{
                  .err = err,
                  .error_out = request != nullptr ? request->error_out : nullptr,
                  .request = request,
                });
                return;
              } else if constexpr (std::is_same_v<decltype(ev.request), const event::reset *>) {
                const event::reset * request = ev.request;
                process(events::reset_error{
                  .err = err,
                  .error_out = request != nullptr ? request->error_out : nullptr,
                  .request = request,
                });
                return;
              }
            }
          },
      sml::state<failed> + sml::event<events::configure_error> / action::on_configure_error =
          sml::state<ready>,
      sml::state<failed> + sml::event<events::allocate_error> / action::on_allocate_error =
          sml::state<ready>,
      sml::state<failed> + sml::event<events::release_error> / action::on_release_error =
          sml::state<ready>,
      sml::state<failed> + sml::event<events::reset_error> / action::on_reset_error =
          sml::state<ready>,

      sml::state<ready> + sml::event<sml::_> / action::on_unexpected = sml::state<failed>,
      sml::state<configuring> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<applying_configure> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<configure_done> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<validating_allocate> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<selecting_block> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<ensuring_chunk> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<committing_allocate> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocate_done> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<validating_release> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<merging_release> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<release_done> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<resetting> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reset_done> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  uint64_t alignment() const noexcept { return context_.alignment; }

  uint64_t max_chunk_size() const noexcept { return context_.max_chunk_size; }

  int32_t chunk_count() const noexcept { return context_.chunk_count; }

  uint64_t chunk_max_size(const int32_t chunk) const noexcept {
    if (chunk < 0 || chunk >= context_.chunk_count) return 0;
    return context_.chunks[chunk].max_size;
  }

 private:
  action::context context_{};
};

}  // namespace emel::buffer::chunk_allocator
