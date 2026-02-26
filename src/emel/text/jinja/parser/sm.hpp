#pragma once

/*
design doc: docs/designs/text/jinja/parser.design.md
 ---
 title: jinja/parser architecture design
 status: draft
 ---
 
 # jinja/parser architecture design
 
 this document defines the jinja parser actor. it replaces the legacy procedural recursive-descent parser with a hierarchical `boost::sml` state machine. because parsing is a one-time initialization step (e.g. during model loading), it utilizes standard heap allocation (`std::make_unique` and `std::vector`) for building the AST, adhering to the rule that one-time initialization may allocate.
 
 ## role
 - act as a pure SML actor that consumes a stream of `jinja::token` objects.
 - build an abstract syntax tree (`program`) using standard C++ heap allocations for AST nodes.
 - enforce strict grammatical rules and nesting bounds using SML submachines and context stacks, eliminating procedural call stack recursion.
 
 ## architecture shift: token-pump and hierarchical SML
 in the legacy design, parsing was a single procedural action (`run_parse`) that invoked a `recursive_descent_parser`. this violated the spirit of the actor model (by hiding complex control flow in an action) and relied heavily on the C++ call stack.
 
 the new architecture uses the SML state machine as a true **Pushdown Automaton**:
 1. **token events**: the external orchestrator (e.g. `text/conditioner`) calls the lexer to get an array of tokens, then iterates over them, dispatching a `token_event` for each one into the parser SM.
 2. **hierarchical machines**: grammatical structures (like `expression` vs `statement`) are modeled as `boost::sml` composite states. transitioning into an expression simply pushes the SM into a submachine, naturally tracking nesting without C++ recursion.
 3. **heap allocation**: since parsing only happens once during setup, actions construct AST nodes dynamically using `std::make_unique` and `std::vector`. this provides full flexibility and simplifies cleanup without violating inference hot-path constraints.
 
 ## events
 - `event::start_parse`
   - inputs: `program_out` (destination AST) and optional callbacks.
   - outputs: initializes the parser's context stacks and transitions from `uninitialized` to `idle`.
 - `event::token_received`
   - inputs: a single `jinja::token` and its metadata (type, string value, offset).
   - outputs: drives the grammar state machine. creates AST nodes on the heap, pushes/pops the parsing stack, or sets a syntax error.
 - `event::end_parse`
   - inputs: none. signals the end of the token stream.
   - outputs: validates that all parsing stacks are empty (no unclosed tags) and finalizes the `program` structure. emits `parsing_done` or `parsing_error`.
 
 ## state model (hierarchical)
 
 ```text
 uninitialized ──► idle (top-level text context)
                    │
                    ├──► (receive `{{`) ──► sm<expression_parser> ──► (receive `}}`) ──► idle
                    │
                    └──► (receive `{%`) ──► sm<statement_parser> ──► (receive `%}`) ──► idle
 ```
 
 ### submachine: `expression_parser`
 handles the Pratt-style operator precedence and function calls.
 
 ```text
 expecting_operand ──► (receive Identifier) ──► expecting_operator
                     ├──► (receive Numeric) ──► expecting_operator
                     └──► (receive `(`) ──► sm<expression_parser> (nested)
 
 expecting_operator ──► (receive `+`) ──► expecting_operand
                      ├──► (receive `(`) ──► sm<call_args_parser>
                      └──► (receive `]`) ──► (pop stack)
 ```
 
 ## responsibilities
 
 1. **one-time heap allocation**:
    - AST nodes (like `binary_expression` or `identifier`) are dynamically allocated via `std::make_unique`.
    - `ast_list` uses `std::vector<std::unique_ptr<ast_node>>`.
    - memory is released naturally when the `program` struct is destroyed.
    
 2. **explicit bounds**:
    - the SML context must track `stack_depth`. if an expression nests too deeply (e.g. `(((((...))))`), the `push_stack` action must trigger a transition to `errored` to prevent malicious nesting attacks.
 
 3. **deterministic error recovery**:
    - if a syntax error occurs (e.g. an unexpected token type for the current state), the machine immediately transitions to `errored` and sets `error_pos` in the context. subsequent `token_received` events are ignored or result in `sml::unexpected_event` routing.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_TEMPLATE_SYNTAX` — unexpected token or malformed grammar construct.
 - `EMEL_ERR_TEMPLATE_UNSUPPORTED` — valid syntax for a construct that is not supported by this implementation.
 - `EMEL_ERR_CAPACITY` — node arena or parsing stack capacity exceeded.
 - `EMEL_ERR_INVALID_ARGUMENT` — invalid token stream, buffer pointers, or limit values.
*/


/*
design doc: docs/designs/text/jinja/lexer.design.md
 ---
 title: jinja/lexer architecture design
 status: draft
 ---
 
 # jinja/lexer architecture design
 
 this document defines the jinja lexer actor. it replaces the legacy procedural `while`-loop tokenizer with a pure `boost::sml` state machine, acting as a character-pump automaton to cleanly identify boundaries, handle escapes, and emit tokens without complex nested loops or procedural lookahead spaghetti.
 
 ## role
 - act as a pure SML actor that consumes a stream of `char` (or `event::char_received`).
 - emit a sequence of `jinja::token` objects via synchronous callbacks or by accumulating them in a context array.
 - gracefully handle malformed templates (e.g., unterminated strings) using strict state transitions rather than complex procedural bounds checking.
 
 ## architecture shift: the character-pump automaton
 in the legacy design, the lexer was a procedural class that ran a massive `while` loop, manually checking indices (`pos + 1`, `pos + 2`) and advancing pointers. this is fragile and violates the actor model paradigm.
 
 the new architecture turns the lexer into a true **Finite State Machine**:
 1. **character events**: the orchestrator iterates over the `std::string_view` of the template and dispatches an `event::char_received` for each character (plus an `event::eof` at the end).
 2. **state-driven tokenization**: the lexer starts in a `text` state. if it receives `{`, it transitions to a `potential_block_open` state. if the next char is `%`, it emits a `text` token (if any was accumulated) and transitions to the `in_statement` state.
 3. **zero-allocation tracking**: the context holds a fixed-size `scratch_buffer` for accumulating the current token's characters (or simply tracks the `start_pos` and `current_pos` indices to yield `std::string_view` tokens).
 
 ## events
 - `event::start_lex`
   - inputs: optional synchronous callbacks for emitting tokens.
   - outputs: initializes context pointers and transitions to `idle`.
 - `event::char_received`
   - inputs: `char c`, `size_t pos`.
   - outputs: drives the state machine. may accumulate characters, transition states, or emit completed tokens.
 - `event::eof`
   - inputs: none.
   - outputs: forces the emission of the final token (if valid). if the machine is in an unterminated state (like `in_string`), it transitions to `errored`.
 
 ## state model
 
 ```text
 uninitialized ──► idle (accumulating text)
                     │
                     ├──► (receive `{`) ──► potential_open
                     │                        ├──► (receive `%`) ──► in_statement
                     │                        ├──► (receive `{`) ──► in_expression
                     │                        └──► (receive `#`) ──► in_comment
                     │
                     └──► (receive EOF) ──► done
 ```
 
 ### sub-states (e.g., `in_expression`)
 when inside an expression or statement, the machine parses numbers, identifiers, strings, and operators:
 
 ```text
 in_expression ──► (receive `"`) ──► in_string ──► (receive `"`) ──► in_expression
               ├──► (receive digit) ──► in_number ──► (receive non-digit) ──► in_expression
               ├──► (receive letter) ──► in_identifier ──► (receive non-word) ──► in_expression
               └──► (receive `}`) ──► potential_close ──► (receive `}`) ──► idle
 ```
 
 ## responsibilities & constraints
 
 1. **index-based string views**:
    - to avoid heap-allocating `std::string` for every token, the lexer context should merely track the `start_pos` of the current token.
    - when a token is complete, it emits a `jinja::token` containing the `token_type` and the `std::string_view` derived from the original template string.
 
 2. **escape sequence handling**:
    - when inside `in_string`, receiving `\\` transitions to an `in_escape` state. the next character is decoded and the machine returns to `in_string`. if `EOF` is received during `in_escape` or `in_string`, the lexer transitions to `errored`.
 
 3. **deterministic error recovery**:
    - invalid syntax (e.g., an illegal character in a variable name or an unterminated string) causes an immediate transition to `errored`.
    - the error position is recorded in the context, and subsequent `char_received` events are ignored via `unexpected_event` routing.
*/


/*
design doc: docs/designs/jinja.design.md
 ---
 title: jinja architecture design
 status: draft
 ---
 
 # jinja architecture design
 
 this document defines the jinja templating subsystem. it provides a high-performance, allocation-bounded jinja2 template engine used for chat formatting and prompt construction within the `text/formatter` and `text/conditioner` pipelines.
 
 ## role
 - provide a lightweight, dependency-free jinja2 implementation tailored for LLM chat templates.
 - execute entirely within the run-to-completion (RTC) actor model.
 - enforce strict memory and computation bounds during both parsing and formatting to prevent denial-of-service from malicious templates.
 
 ## architecture: explicit decoupling
 the jinja subsystem is divided into three distinct phases to ensure strict separation of concerns and allow offline compilation of templates in the future:
 1. **`lexer`**: a pure, synchronous utility that converts a raw template string into a bounded array of tokens.
 2. **`parser::sm`**: an SML actor that consumes a token stream and builds an abstract syntax tree (`program`).
 3. **`formatter::sm`**: an SML actor that evaluates a `program` against a set of `globals` and emits formatted UTF-8 bytes.
 
 ## lexer
 the lexer is a pure utility (not an SML actor) that takes a `std::string_view` and returns a `lexer_result` containing a sequence of tokens.
 - **responsibilities**: handle whitespace trimming (`{%-`, `-%}`), detect blocks, comments, and expressions, and reject invalid syntax (like unterminated strings).
 - **invariants**: operates sequentially without heap allocating individual tokens, typically returning a flat array/vector of bounded tokens.
 
 ## parser (parser::sm)
 the parser is an SML actor that converts the token stream into a structured `program` (AST).
 
 ### events
 - `event::parse`
   - inputs: `template_text`, `program_out` (AST destination), `error_out`, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: tokenizes and parses the template text, populates the provided `program`, and invokes the appropriate callback before returning.
 
 ### state model
 ```text
 uninitialized ──► initialized
                       │
 initialized ──► parse_decision ──► (done | errored)
   ▲                                    │
   └────────────────────────────────────┘
 ```
 - `initialized` — awaiting parse intent.
 - `parse_decision` — synchronously runs the internal parsing logic (via recursive descent) and branches based on success/failure.
 - `done` — parsing complete and AST populated.
 - `errored` — syntax error or invalid argument.
 - unexpected events route to `unexpected`.
 
 ## formatter (formatter::sm)
 the formatter is an SML actor that evaluates a parsed `program` and emits the final formatted text. it supports variables, control flow (if/for/macro), filters, and tests.
 
 ### events
 - `event::format`
   - inputs: `program` (parsed AST), `globals` (object containing user variables and model metadata), caller-provided `output` buffer + capacity, `output_length`, `error_out`, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: evaluates the AST, writes UTF-8 bytes directly to the `output` buffer, updates `output_length`, and invokes the callback.
 
 ### state model
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
 - `initialized` — idle state awaiting format intent.
 - `setup` — initializes the evaluation context and seeds the initial statement work.
 - `eval_stmt` — steps through statements in the current scope.
 - `eval_expr` — evaluates pending expressions (variable resolution, math, filters).
 - `write_output` — emits the evaluated values to the output buffer.
 - `format_decision` — checks if the evaluation completed successfully or hit an error/limit.
 - `done` — formatting complete.
 - `errored` — hit an execution limit (e.g., max scope depth, out of buffer capacity, division by zero) or invalid arguments.
 - unexpected events route to `unexpected`.
 
 ### responsibilities & constraints
 1. **bounded evaluation**: the formatter strictly limits evaluation depth, scope counts, call stack limits, and array/object sizes to prevent infinite loops or memory exhaustion (enforced by constants like `k_max_scopes`, `k_max_array_items`, `k_max_callables`).
 2. **zero-allocation formatting**: outputs are written directly to caller-provided buffers (`format_io`). intermediate strings are not dynamically heap-allocated during template evaluation; they are either string views of the original template or written into a fixed-size scratch buffer in the SML context.
 3. **synchronous callbacks**: results are communicated via `dispatch_done` and `dispatch_error` callbacks before the SML dispatch returns, eliminating the need for the caller to inspect the state machine's internal context.
*/


#include <cstdint>

#include "emel/text/jinja/parser/actions.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::text::jinja::parser {

struct initialized {};
struct parse_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * jinja parser orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting parse intent.
 * - `parse_decision`: run parsing step and branch based on phase results.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_parse`/`invalid_parse` validate request pointers and parameters.
 * - `phase_*` guards observe errors set by actions.
 *
 * action side effects:
 * - `run_parse` returns format unsupported until parser is implemented.
 * - `on_unexpected` reports any event sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<initialized> +
            sml::event<event::parse>[guard::valid_parse{}] /
                action::run_parse = sml::state<parse_decision>,
        sml::state<initialized> +
            sml::event<event::parse>[guard::invalid_parse{}] /
                action::reject_invalid_parse = sml::state<errored>,

        sml::state<done> + sml::event<event::parse>[guard::valid_parse{}] /
                               action::run_parse = sml::state<parse_decision>,
        sml::state<done> + sml::event<event::parse>[guard::invalid_parse{}] /
                               action::reject_invalid_parse =
            sml::state<errored>,

        sml::state<errored> + sml::event<event::parse>[guard::valid_parse{}] /
                                  action::run_parse = sml::state<parse_decision>,
        sml::state<errored> + sml::event<event::parse>[guard::invalid_parse{}] /
                                  action::reject_invalid_parse =
            sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::parse>[guard::valid_parse{}] /
                action::run_parse = sml::state<parse_decision>,
        sml::state<unexpected> +
            sml::event<event::parse>[guard::invalid_parse{}] /
                action::reject_invalid_parse = sml::state<unexpected>,

        sml::state<parse_decision>[guard::phase_ok{}] =
            sml::state<done>,
        sml::state<parse_decision>[guard::phase_failed{}] =
            sml::state<errored>,

        sml::state<initialized> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<parse_decision> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<done> + sml::unexpected_event<sml::_> /
                               action::on_unexpected = sml::state<unexpected>,
        sml::state<errored> + sml::unexpected_event<sml::_> /
                                  action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> +
            sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;
  using base_type::base_type;
  using base_type::process_event;
};

} // namespace emel::text::jinja::parser
