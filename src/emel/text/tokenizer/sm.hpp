#pragma once

/*
design doc: docs/designs/text/unicode.design.md
 ---
 title: text/unicode architecture design
 status: draft
 ---
 
 # text/unicode architecture design
 
 this document defines the text/unicode subsystem. it provides a high-performance, zero-allocation utility for decoding, encoding, normalizing, and categorizing utf-8 strings into unicode code points, which is a prerequisite for accurate tokenization.
 
 ## role
 - act as a pure, stateless utility library (not an SML actor) that provides unicode operations to the `text/tokenizer/preprocessor` and `text/encoders` actors.
 - provide unicode category classification (e.g., `is_letter`, `is_whitespace`, `is_punctuation`) using compacted, binary-searchable range tables.
 - perform unicode normalization (e.g., NFD) and case folding (e.g., lowercase mapping).
 - handle highly optimized, model-specific unicode regex splitting (e.g., LLaMA-3, GPT-2) without relying on heavy and slow `std::regex` engines.
 
 ## architecture shift: pure stateless utilities and table compaction
 in legacy systems (like `llama.cpp`), unicode operations often relied on massive, dense lookup tables or dragged in heavyweight regex engines that allocated heavily on the heap and hurt performance in the hot path.
 
 in `emel`, the unicode subsystem strictly adheres to the performance and zero-allocation invariants:
 
 1. **not an SML actor**: unicode normalization and querying is mathematically pure. it has no internal state across calls, no lifecycle, and no side-effects. modeling it as an SML actor would introduce unnecessary dispatch overhead. it is a library of static utility functions.
 2. **compacted range tables**: instead of a flat array of 1.1 million `uint16_t` flags (which blows out the CPU cache), unicode categories and normalization maps are stored as contiguous range pairs (`[start_codepoint, end_codepoint, flags]`). looking up a character's category is a fast `O(log N)` binary search over a very small, cache-friendly array.
 3. **procedural regex fallbacks**: rather than using `std::regex` to split input text (which allocates memory and is notoriously slow in C++), `emel` provides hand-rolled, zero-allocation procedural state machines for the specific regexes used by major models (e.g., `unicode_regex_split_custom_llama3`). these functions iterate over the string in a single pass, yielding token boundaries into a pre-allocated array.
 
 ## core capabilities
 
 ### 1. utf-8 conversion
 - `unicode_cpt_from_utf8(std::string_view, size_t& pos)`: decodes a single utf-8 character into a `uint32_t` code point, advancing the position.
 - `unicode_cpt_to_utf8(uint32_t)`: encodes a code point back to utf-8.
 
 ### 2. categorization (`unicode_cpt_flags`)
 provides a bitmask for any given code point:
 - `is_letter` (`\p{L}`)
 - `is_number` (`\p{N}`)
 - `is_punctuation` (`\p{P}`)
 - `is_whitespace`
 - `is_lowercase` / `is_uppercase`
 
 ### 3. normalization
 - `unicode_tolower(uint32_t)`: applies unicode lowercase folding.
 - `unicode_normalize_nfd(const std::vector<uint32_t>&)`: applies canonical decomposition.
 
 ### 4. regex pre-splitting
 - `unicode_regex_split_custom(...)`: given a specific model's regex pattern (e.g., GPT-2's `'s|'t|...`), this function bypasses `std::regex` and uses a hardcoded, highly optimized loop to find split boundaries.
 
 ## integration with tokenization
 the `text/tokenizer/preprocessor::sm` uses the unicode subsystem during its `preprocessing` phase. for example, when a BPE preprocessor needs to split a string into fragments based on LLaMA-3's rules, it calls the `unicode_regex_split_custom_llama3` utility, converting the boundaries into SML context fragments. the unicode subsystem itself remains entirely oblivious to the SML orchestration or the tokenization lifecycle.
*/


/*
design doc: docs/designs/text/tokenizer.design.md
 ---
 title: text/tokenizer architecture design
 status: rolling
 ---
 
 # text/tokenizer architecture design
 
 this document captures the tokenizer actor as implemented today. it reflects the current
 orchestration model and data contracts; any structural changes still require explicit approval.
 
 ## role
 - text/tokenizer is a codec-stage actor: text -> token ids.
 - tokenizer composes a preprocessor + `text/encoders::any` pair selected from model metadata.
 
 ## public interface
 - `event::bind` (vocab binding)
   - inputs: `vocab`, `error_out`, optional sync callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: `error_out` and callbacks; tokenizer transitions to `idle` on success.
 - `event::tokenize` (encode text)
   - inputs: `vocab`, `text`, `add_special`, `parse_special`, `token_ids_out`,
     `token_capacity`, `token_count_out`, `error_out`, optional callbacks.
   - outputs: `token_count_out`, `error_out`, and done/error callbacks.
 
 callbacks (`dispatch_done`/`dispatch_error`) are invoked synchronously before dispatch returns and
 are not stored in context.
 
 ## composition and explicit dependency injection (DI)
 to maintain strict modularity and fast compile times, the `text/tokenizer` relies on **explicit
 dependency injection**. it does not parse model metadata to auto-instantiate its sub-components.
 
 - **owns (injected SML actors):**
   - `text/tokenizer::preprocessor::any` (variant preprocessor SM, e.g., using `emel::sm_any`).
   - `text/encoders::any` (variant encoder SM, e.g., using `emel::sm_any`).
 - **binding:** the `preprocessor` and `encoder` variants (e.g., `bpe`, `spm`, `rwkv`) MUST be explicitly
   injected by the caller (like the `text/conditioner` or a higher-level factory) via the `event::bind`
   payload. the tokenizer does not instantiate them internally based on model metadata.
 
## architecture shift: output via synchronous callbacks
to strictly enforce the Actor Model isolation and prevent future race conditions, the
`text/tokenizer` relies heavily on **synchronous callbacks** rather than exposing read-only snapshots
of its internal SML context.
 
 as defined in `sml.rules.md`, events carry `emel::callback`-style functors (e.g., `dispatch_done`,
 `dispatch_error`). the tokenizer guarantees these callbacks are invoked *before* the SML dispatch
 returns. this ensures the caller immediately receives the results (or errors) without needing to hold
 a reference to the tokenizer's internal state machine context, completely eliminating the risk of
 read/write race conditions across concurrent steps.
 
 ## state model (current)
 - bind flow:
   `uninitialized` -> `binding_preprocessor` -> `binding_preprocessor_decision`
   -> `binding_encoder` -> `binding_encoder_decision` -> `idle`.
 - tokenization flow:
   `idle` -> `preprocessing` -> `preprocess_decision` -> `prefix_decision`
   -> `encoding_ready` -> (`encoding_token_fragment` | `encoding_raw_fragment`)
   -> `encoding_decision` -> `encoding_ready` (loop)
   -> `suffix_decision` -> `finalizing` -> `done`.
 - errors route to `errored`; sequencing violations go to `unexpected`.
 
 ## preprocessing
 - preprocessor emits a bounded fragment list (`k_max_fragments`) into context.
 - fragments are either `token` (pre-resolved special token id) or `raw_text` spans.
 - `parse_special` controls whether special tokens are extracted in preprocessing.
 
 ## encoding
 - encoding iterates fragments in a bounded RTC loop.
 - `token` fragments append directly to `token_ids_out`.
 - `raw_text` fragments are encoded by the bound encoder via `text/encoders::any`.
 - `preprocessed` flag is forwarded to the encoder.
 
 ## prefix/suffix rules (current)
 - BOS: added when `add_special && vocab->add_bos`.
 - SEP: added when `add_special && model_kind == wpm && vocab->add_sep`.
 - EOS: added when `add_special && model_kind != wpm && vocab->add_eos`.
 - prefix/suffix additions require capacity and valid ids.
 
 ## error mapping
 - invalid request, capacity overflow, or unexpected events -> `EMEL_ERR_INVALID_ARGUMENT`.
 - invalid vocab ids -> `EMEL_ERR_MODEL_INVALID`.
 - kernel/preprocessor/encoder failures propagate their `error_out` values.
 
 ## invariants
 - no allocation during dispatch; fragments live in fixed-size arrays.
 - no self-dispatch; internal progress uses anonymous transitions.
 - outputs are written only through request payloads (no persistent output buffers).
 
 ## tests (current)
 - tokenizer orchestration tests for bind, prefix/suffix, capacity, and error paths.
 - per-encoder and per-preprocessor unit tests.
*/


#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/text/tokenizer/actions.hpp"
#include "emel/text/tokenizer/events.hpp"
#include "emel/text/tokenizer/guards.hpp"

namespace emel::text::tokenizer {

struct uninitialized {};
struct binding_preprocessor {};
struct binding_preprocessor_decision {};
struct binding_encoder {};
struct binding_encoder_decision {};
struct idle {};
struct preprocessing {};
struct preprocess_decision {};
struct prefix_decision {};
struct encoding_ready {};
struct encoding_token_fragment {};
struct encoding_raw_fragment {};
struct encoding_decision {};
struct suffix_decision {};
struct finalizing {};
struct done {};
struct errored {};
struct unexpected {};

/*
tokenizer architecture notes (single source of truth)

scope
- component boundary: tokenizer
- goal: tokenize text into token ids with special-token partitioning and
model-aware encoding.

state purpose
  - uninitialized: no bound vocab, awaits bind.
  - binding_preprocessor/binding_encoder: bind model-specific preprocess/encode stages.
  - idle: ready to tokenize requests.
  - preprocessing: dispatch preprocessor to build fragment list.
  - preprocess_decision: routes based on preprocess success/failure.
  - prefix_decision: applies optional BOS prefix or errors.
  - encoding_ready/encoding_*: encodes fragments in a bounded loop.
  - suffix_decision: applies optional SEP/EOS suffix or errors.
  - finalizing: marks success.
  - done: last request completed successfully.
- errored: last request failed with an error code.
- unexpected: sequencing contract violation.

key invariants
- per-request outputs are written only through the triggering event payload.
- context owns only runtime data (fragments, encoder context, counters).
- internal progress uses anonymous transitions (no self-dispatch).

guard semantics
  - can_bind: validates bind request pointers.
  - can_tokenize: validates request pointers, capacity, and bound vocab match.
  - phase_ok/phase_failed: observe errors set by actions.
  - has_capacity: checks remaining output capacity before encoding.
  - should_add_bos/sep/eos: determines prefix/suffix requirements.
  - has_more_fragments: indicates more fragments to encode.

action side effects
  - begin_bind: stores vocab and resets bind error state.
  - bind_preprocessor/bind_encoder: select backend machines for model.
  - begin_tokenize: resets request outputs and context runtime state.
  - run_preprocess: builds fragment list, honoring parse_special.
  - append_bos/sep/eos: appends prefix/suffix tokens as configured by vocab.
  - append_fragment_token/encode_raw_fragment: encode a fragment or append a
  literal token.
- set_capacity_error/set_invalid_id_error: records validation failures.
- finalize: marks success.
- on_unexpected: reports sequencing violations.
*/
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<uninitialized> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<uninitialized> + sml::event<event::bind> /
                action::reject_bind = sml::state<errored>,
        sml::state<uninitialized> + sml::event<event::tokenize> /
                action::reject_invalid = sml::state<errored>,

        sml::state<binding_preprocessor> / action::bind_preprocessor =
            sml::state<binding_preprocessor_decision>,
        sml::state<binding_preprocessor_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<binding_preprocessor_decision>[guard::phase_ok{}] =
            sml::state<binding_encoder>,

        sml::state<binding_encoder> / action::bind_encoder =
            sml::state<binding_encoder_decision>,
        sml::state<binding_encoder_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<binding_encoder_decision>[guard::phase_ok{}] =
            sml::state<idle>,

        sml::state<idle> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<idle> + sml::event<event::bind> /
                action::reject_bind = sml::state<errored>,
        sml::state<idle> + sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<preprocessing>,
        sml::state<idle> + sml::event<event::tokenize> /
                action::reject_invalid = sml::state<errored>,

        sml::state<done> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<done> + sml::event<event::bind> /
                action::reject_bind = sml::state<errored>,
        sml::state<done> + sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<preprocessing>,
        sml::state<done> + sml::event<event::tokenize> /
                action::reject_invalid = sml::state<errored>,

        sml::state<errored> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<errored> + sml::event<event::bind> /
                action::reject_bind = sml::state<errored>,
        sml::state<errored> +
            sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<preprocessing>,
        sml::state<errored> + sml::event<event::tokenize> /
                action::reject_invalid = sml::state<errored>,

        sml::state<unexpected> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<unexpected> + sml::event<event::bind> /
                action::reject_bind = sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<preprocessing>,
        sml::state<unexpected> +
            sml::event<event::tokenize> / action::reject_invalid =
            sml::state<unexpected>,

        sml::state<preprocessing> / action::run_preprocess =
            sml::state<preprocess_decision>,
        sml::state<preprocess_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<preprocess_decision>[guard::phase_ok{}] =
            sml::state<prefix_decision>,

        sml::state<prefix_decision>[guard::bos_ready{}] /
            action::append_bos = sml::state<encoding_ready>,
        sml::state<prefix_decision>[guard::bos_no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<prefix_decision>[guard::bos_invalid_id{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<prefix_decision>[guard::no_prefix{}] =
            sml::state<encoding_ready>,

        sml::state<encoding_ready>[guard::no_more_fragments{}] =
            sml::state<suffix_decision>,
        sml::state<encoding_ready>[guard::more_fragments_no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<encoding_ready>[guard::more_fragments_token{}] =
            sml::state<encoding_token_fragment>,
        sml::state<encoding_ready>[guard::more_fragments_raw{}] =
            sml::state<encoding_raw_fragment>,

        sml::state<encoding_token_fragment> / action::append_fragment_token =
            sml::state<encoding_decision>,
        sml::state<encoding_raw_fragment> / action::encode_raw_fragment =
            sml::state<encoding_decision>,
        sml::state<encoding_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<encoding_decision>[guard::phase_ok{}] =
            sml::state<encoding_ready>,

        sml::state<suffix_decision>[guard::sep_ready{}] /
            action::append_sep = sml::state<finalizing>,
        sml::state<suffix_decision>[guard::sep_no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::sep_invalid_id{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::eos_ready{}] /
            action::append_eos = sml::state<finalizing>,
        sml::state<suffix_decision>[guard::eos_no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::eos_invalid_id{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::no_suffix{}] =
            sml::state<finalizing>,

        sml::state<finalizing> / action::finalize = sml::state<done>,

        sml::state<uninitialized> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<binding_preprocessor> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<binding_preprocessor_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<binding_encoder> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<binding_encoder_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<idle> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<preprocessing> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<preprocess_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<prefix_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding_ready> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding_token_fragment> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding_raw_fragment> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<suffix_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<finalizing> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<done> + sml::unexpected_event<sml::_> /
                               action::on_unexpected = sml::state<unexpected>,
        sml::state<errored> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() : base_type() {}

  bool process_event(const event::bind &ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<idle>);
    const int32_t err =
        ok ? EMEL_OK
           : (this->context_.last_error != EMEL_OK ? this->context_.last_error
                                             : EMEL_ERR_BACKEND);

    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::tokenizer_bind_done{&ev});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::tokenizer_bind_error{&ev, err});
      }
    }

    action::clear_request(this->context_);
    return accepted && ok;
  }

  bool process_event(const event::tokenize &ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err =
        ok ? EMEL_OK
           : (this->context_.last_error != EMEL_OK ? this->context_.last_error
                                             : EMEL_ERR_BACKEND);

    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = this->context_.token_count;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm,
                         events::tokenizer_done{&ev, this->context_.token_count});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::tokenizer_error{&ev, err});
      }
    }

    action::clear_request(this->context_);
    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return this->context_.last_error; }
  int32_t token_count() const noexcept { return this->context_.token_count; }

private:
};

} // namespace emel::text::tokenizer
