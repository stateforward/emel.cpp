#pragma once

/*
design doc: docs/designs/text/jinja/formatter.design.md
 ---
 title: jinja/formatter architecture design
 status: draft
 ---
 
 # jinja/formatter architecture design
 
 this document defines the jinja formatter actor (formerly `renderer::sm`). it evaluates a parsed `jinja::program` against a set of variables and emits formatted text. it enforces strict memory and computation bounds using statically provable limits (e.g., stack depth, scope counts) to ensure safe, zero-allocation execution within the inference hot path.
 
 ## role
 - act as a pure SML actor that evaluates a Jinja AST (`program`).
 - emit UTF-8 formatted text into caller-provided buffers (`format_io`) without heap allocation.
 - prevent denial-of-service (DoS) from malicious or infinitely recursive templates by enforcing strict execution bounds.
 
 ## architecture shift: renaming and bounding
 previously named the `renderer`, this component has been renamed to `formatter` to align with its role within the `text/conditioner` pipeline (where a `text/formatter` prepares prompt strings before tokenization, while a `text/renderer` is responsible for decoding tokens back into domain output).
 
 the core architectural challenge in the formatter is its evaluation loop:
 ```text
 eval_stmt ──► eval_expr ──► write_output ──► eval_stmt
 ```
 because these are **anonymous transitions** (transitions taken automatically without an external event), SML's Run-To-Completion (RTC) semantics will loop them until the machine reaches quiescence. if a template contains an infinite loop or unbounded recursion (e.g., `{% macro inf() %}{{ infinite() }}{% endmacro %}{{ inf() }}`), the SML dispatch would never return, locking the thread.
 
 ### the stack depth solution
 to satisfy the `sml.rules.md` requirement that *"anonymous transition graphs MUST be acyclic or MUST have a statically provable bound on firings per top-level event"*, the formatter enforces strict hierarchical limits rather than a flat instruction counter.
 
 1. **call stack depth**: the context tracks `call_depth`. every time a macro or function is evaluated, the depth increments. if it exceeds `k_max_call_depth`, the action sets an error flag.
 2. **scope bounds**: the context tracks `scope_count`. every `{% for %}` loop or block pushes a new variable scope. if this exceeds `k_max_scopes`, it errors out.
 3. **step limits (fail-safe)**: as a final safeguard against massive (but not necessarily deep) templates, a `steps_remaining` counter is decremented on every statement evaluation. if it hits 0, execution halts.
 
 these limits ensure the anonymous SML loops are mathematically guaranteed to terminate and return control to the orchestrator.
 
 ## events
 - `event::format`
   - inputs: `program` (the parsed AST), `globals` (injected variables/context), `output` buffer pointer and `output_capacity`, `error_out`, and optional synchronous callbacks.
   - outputs: executes the AST, populates the `output` buffer, updates `output_length` and `output_truncated` flags, and invokes the callback before returning.
 
 ## state model
 
 ```text
 uninitialized ──► initialized
                       │
 initialized ──► setup ──► eval_stmt ──► format_decision ──► (done | errored)
   ▲                         │  ▲                                │
   │                         ▼  │                                │
   │                     eval_expr ──► write_output              │
   │                                     │                       │
   └─────────────────────────────────────┴───────────────────────┘
 ```
 
 - `setup` — binds the `program`, initializes the scope stack and bounds counters, and prepares the initial statements.
 - `eval_stmt` — evaluates control flow (`if`, `for`, `set`) and enqueues expression evaluation.
 - `eval_expr` — resolves variables, performs math, applies filters, and executes macros.
 - `write_output` — writes evaluated expression values into the caller's output buffer.
 - `format_decision` — inspects the context's error state (e.g., bounds exceeded, invalid types, buffer full).
 - `done` / `errored` — terminal states for the RTC chain.
 
 ## responsibilities & constraints
 
 1. **zero-allocation rendering**:
    - the formatter must never use `new`, `malloc`, or `std::string` during evaluation.
    - temporary strings (like the result of `replace` or `join` filters) are stored in a fixed-size `string_buffer` within the SML context. if this buffer fills up (`k_max_string_bytes`), the formatter immediately errors.
    - output is written sequentially to the caller-provided `format_io` buffer.
 
 2. **graceful truncation**:
    - if the evaluated text exceeds the caller's `output_capacity`, the formatter must not crash. it stops writing, sets `output_truncated = true` in the event payload, and gracefully finishes the evaluation or errors out depending on the strictness policy.
 
 3. **deterministic error routing**:
    - any bounds violation (stack depth, scope count, step limit) or runtime error (division by zero, missing filter) sets `ctx.phase_error`. the state machine guards inspect this and cleanly route to `format_decision` -> `errored` without throwing C++ exceptions.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_TEMPLATE_RUNTIME` — runtime evaluation error (e.g., division by zero, missing filter, invalid type).
 - `EMEL_ERR_TEMPLATE_LIMIT` — execution limit exceeded (call stack depth, scope count, or step count).
 - `EMEL_ERR_CAPACITY` — output buffer full or string buffer exhausted.
 - `EMEL_ERR_INVALID_ARGUMENT` — invalid program, globals, or buffer pointers.
*/


#include <cstdint>

#include "emel/text/jinja/formatter/actions.hpp"
#include "emel/text/jinja/formatter/events.hpp"
#include "emel/text/jinja/formatter/guards.hpp"
#include "emel/sm.hpp"

namespace emel::text::jinja::formatter {

struct initialized {};
struct setup {};
struct eval_stmt {};
struct eval_expr {};
struct write_output {};
struct render_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * jinja renderer orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting render intent.
 * - `setup`: initialize context and seed statement work.
 * - `eval_stmt`: step through statements and enqueue expressions when needed.
 * - `eval_expr`: evaluate pending expressions.
 * - `write_output`: emit pending expression values.
 * - `render_decision`: branch based on phase results.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_render`/`invalid_render` validate request pointers and parameters.
 * - `phase_*` guards observe errors set by actions.
 *
 * action side effects:
 * - `begin_render`/`seed_program` prepare context for a render pass.
 * - `eval_next_stmt`/`eval_pending_expr`/`write_pending_value` execute rendering steps.
 * - `finalize_*` finalize terminal status on context.
 * - `reject_invalid_render` writes errors for invalid requests.
 * - `on_unexpected` reports sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<initialized> +
            sml::event<event::render>[guard::valid_render] / action::begin_render =
                sml::state<setup>,
        sml::state<initialized> +
            sml::event<event::render>[guard::invalid_render] /
                action::reject_invalid_render = sml::state<errored>,

        sml::state<done> + sml::event<event::render>[guard::valid_render] /
                               action::begin_render = sml::state<setup>,
        sml::state<done> + sml::event<event::render>[guard::invalid_render] /
                               action::reject_invalid_render =
            sml::state<errored>,

        sml::state<errored> + sml::event<event::render>[guard::valid_render] /
                                  action::begin_render = sml::state<setup>,
        sml::state<errored> + sml::event<event::render>[guard::invalid_render] /
                                  action::reject_invalid_render =
            sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::render>[guard::valid_render] / action::begin_render =
                sml::state<setup>,
        sml::state<unexpected> +
            sml::event<event::render>[guard::invalid_render] /
                action::reject_invalid_render = sml::state<unexpected>,

        sml::state<setup> / action::seed_program = sml::state<eval_stmt>,

        sml::state<eval_stmt>[guard::phase_failed{}] = sml::state<render_decision>,
        sml::state<eval_stmt>[guard::needs_expr{}] = sml::state<eval_expr>,
        sml::state<eval_stmt>[guard::has_stmt_work{}] / action::eval_next_stmt =
            sml::state<eval_stmt>,
        sml::state<eval_stmt>[guard::no_stmt_work{}] = sml::state<render_decision>,

        sml::state<eval_expr>[guard::has_pending_expr{}] / action::eval_pending_expr =
            sml::state<eval_expr>,
        sml::state<eval_expr>[guard::phase_failed{}] = sml::state<render_decision>,
        sml::state<eval_expr> = sml::state<write_output>,

        sml::state<write_output>[guard::phase_failed{}] = sml::state<render_decision>,
        sml::state<write_output>[guard::needs_write{}] / action::write_pending_value =
            sml::state<write_output>,
        sml::state<write_output> = sml::state<eval_stmt>,

        sml::state<render_decision>[guard::phase_ok{}] / action::finalize_done =
            sml::state<done>,
        sml::state<render_decision>[guard::phase_failed{}] / action::finalize_error =
            sml::state<errored>,

        sml::state<initialized> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<setup> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<eval_stmt> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<eval_expr> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<write_output> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<render_decision> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<done> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<errored> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>,
        sml::state<unexpected> + sml::unexpected_event<sml::_> /
            action::on_unexpected = sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  explicit sm(action::context & ctx) : base_type(ctx), context_(&ctx) {}

  bool process_event(const event::render & ev) {
    namespace sml = boost::sml;

    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const bool valid = guard::valid_render(ev);
    const int32_t err = ok ? EMEL_OK
                           : (context_->last_error != EMEL_OK ? context_->last_error
                                                               : EMEL_ERR_BACKEND);
    const size_t output_length = valid ? context_->output_length : 0;
    const size_t error_pos = valid ? context_->error_pos : 0;
    const bool output_truncated = valid ? (err != EMEL_OK) : false;

    if (ev.output_length != nullptr) {
      *ev.output_length = output_length;
    }
    if (ev.output_truncated != nullptr) {
      *ev.output_truncated = output_truncated;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ev.error_pos_out != nullptr) {
      *ev.error_pos_out = error_pos;
    }

    if (ok) {
      if (ev.dispatch_done) {
        ev.dispatch_done(events::rendering_done{&ev, output_length, output_truncated});
      }
    } else {
      if (ev.dispatch_error) {
        ev.dispatch_error(events::rendering_error{&ev, err, error_pos});
      }
    }

    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

 private:
  action::context * context_ = nullptr;
};

}  // namespace emel::text::jinja::formatter
