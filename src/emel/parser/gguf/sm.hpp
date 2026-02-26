#pragma once

/*
design doc: docs/designs/parser/gguf.design.md
 ---
 title: parser/gguf architecture design
 status: draft
 ---
 
 # parser/gguf architecture design
 
 this document defines the parser/gguf actor. it is responsible for safely decoding the binary GGUF file format, extracting metadata, and providing a structured view of the model's architecture and tensors to the higher-level `model/loader`.
 
 ## role
 - act as a one-time initialization actor that consumes a binary buffer (or file handle) of a GGUF file.
 - parse the GGUF header, key-value pairs (hyper-parameters, architecture, vocabulary), and tensor metadata (name, shape, offset, quantization type).
 - validate the integrity and alignment of the binary data before it is mapped to hardware.
 - facilitate the zero-copy `mmap` of weight data by yielding tensor offsets without copying the actual weights into RAM.
 
 ## architecture shift: decoupled metadata extraction
 in legacy `llama.cpp`, GGUF parsing, memory mapping, and backend buffer allocation were entangled within `llama_model_load`. 
 
 in `emel`, the `parser/gguf` is **strictly a metadata parser**. it does not allocate hardware buffers, it does not execute `mmap` itself, and it does not know about GPU or CPU backends. it simply validates the binary structure and yields a clean, parsed metadata struct to the orchestrating `model/loader`.
 
 ## events
 - `event::probe`
   - inputs: a pointer to the immutable file image and its size.
   - outputs: extracts model requirements from the GGUF header — tensor count, total weight bytes, kv pair count, and maximum key/value byte lengths — without allocating storage. the actor reads only the header and count fields, populates a caller-provided requirements structure, then transitions to `probed`.
 
 - `event::bind_storage`
   - inputs: pre-sized buffers for kv key/value storage arenas, kv entry arrays, and tensor descriptor arrays, along with their capacities.
   - outputs: the actor binds these caller-provided buffers and validates that capacities satisfy the requirements computed during probe. on success, transitions to `bound`. parsing can then proceed.
 
 - `event::parse`
   - inputs: the file image pointer and size (same image used during probe).
   - outputs: parses the full GGUF content — architecture, hyperparameters, vocabulary, and tensor metadata — into the bound storage. populates kv entries and tensor descriptors. transitions to `parsed` on success.
 
 ## state model
 
 the parser follows a three-phase lifecycle driven by external events. the old pattern went directly from uninitialized into parsing; the new pattern separates probing from binding so the caller controls allocation.
 
 ```text
 uninitialized ──► probed ──► bound ──► parsed
                     │          │          │
                     └──────────┴──────────┴──► errored
 ```
 
 - `uninitialized` — idle state awaiting a probe event.
 - `probed` — the header has been read and requirements are available. the actor is waiting for the caller to provide storage via `event::bind_storage`.
 - `bound` — caller-provided buffers are bound and validated. the actor is ready to parse the full file content.
 - `parsed` — parsing complete; kv entries and tensor descriptors are populated in the bound storage. the `model/loader` can now use the metadata to proceed with weight loading and graph assembly.
 - `errored` — hit an invalid GGUF magic, version mismatch, corrupted data offset, or a capacity violation.
 
 within the `bound -> parsed` transition, the actor internally walks through the same logical sub-steps as before (architecture extraction, hparam parsing, vocabulary parsing, tensor mapping), but these are internal to the parse action rather than exposed as separate states.
 
 ## responsibilities & constraints
 
 1. **zero-allocation metadata traversal**:
    - the parser navigates the binary structure using standard pointers and offsets.
    - it does not heap-allocate intermediate strings for keys. it uses `std::string_view` mapped directly over the binary buffer.
 
 2. **mmap delegation**:
    - the parser only computes `file_offset` for each tensor.
    - the higher-level `model/loader` takes this metadata, invokes the OS `mmap`, and uses the `graph/assembler` to create the `leaf` (weight) tensors pointing to those memory-mapped regions.
 
 3. **strict binary validation**:
    - all reads must be strictly bounds-checked against the file size.
    - string lengths and array counts in the GGUF format must not overflow buffer capacities, transitioning to `errored` with `EMEL_ERR_INVALID_ARGUMENT` upon violation.
 
 ## probe-bind lifecycle
 
 the parser uses a two-phase approach to keep allocation entirely outside SML dispatch:
 
 1. **probe** — the caller dispatches `event::probe` with the file image. the actor reads only the GGUF header and count fields, populating a `gguf_requirements` structure with tensor count, kv pair count, maximum key/value byte lengths, and any other sizing information the caller needs. this is a cheap, read-only pass over the first few hundred bytes of the file. the actor transitions to `probed`.
 
 2. **bind** — the caller inspects the requirements, allocates appropriately sized buffers (arenas for key/value bytes, arrays for kv entries and tensor descriptors), and dispatches `event::bind_storage` with pointers and capacities. the actor validates that the provided capacities satisfy the probed requirements, binds the buffers, and transitions to `bound`.
 
 3. **parse** — the caller dispatches `event::parse` with the same file image. the actor walks the full GGUF structure, writing parsed kv entries and tensor descriptors into the bound storage. on completion it transitions to `parsed`.
 
 this pattern means the actor never decides how much memory to allocate or where it comes from. the caller (typically `model/loader`) has full control over allocation strategy — it can use arena allocators, stack buffers, or pre-reserved pools. the parser simply writes into whatever it is given, and rejects buffers that are too small with `EMEL_ERR_CAPACITY`.
 
 recovery from any error requires restarting from probe.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_MODEL_INVALID` — the GGUF header magic or version is invalid, or the file structure is malformed.
 - `EMEL_ERR_CAPACITY` — a buffer provided via `bind_storage` is too small for the probed requirements, or parsed counts exceed configured maxima.
 - `EMEL_ERR_INVALID_ARGUMENT` — string lengths, array counts, or offsets overflow the file buffer or arena capacities.
 - `EMEL_ERR_INTERNAL` — an internal invariant was violated during parsing.
*/


#include "boost/sml.hpp"
#include "emel/parser/gguf/actions.hpp"
#include "emel/parser/gguf/events.hpp"
#include "emel/parser/gguf/guards.hpp"
#include "emel/sm.hpp"

namespace emel::parser::gguf {

struct uninitialized {};
struct probed {};
struct bound {};
struct parsed {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<uninitialized> + sml::event<event::probe> [guard::valid_probe{}] /
        action::run_probe = sml::state<probed>,
      sml::state<uninitialized> + sml::event<event::probe> [guard::invalid_probe{}] /
        action::set_invalid_argument = sml::state<errored>,

      sml::state<probed> + sml::event<event::bind_storage> [guard::valid_bind{}] /
        action::run_bind_storage = sml::state<bound>,
      sml::state<probed> + sml::event<event::bind_storage> [guard::invalid_bind{}] /
        action::set_invalid_argument = sml::state<errored>,

      sml::state<bound> + sml::event<event::parse> [guard::valid_parse{}] /
        action::run_parse = sml::state<parsed>,
      sml::state<bound> + sml::event<event::parse> [guard::invalid_parse{}] /
        action::set_invalid_argument = sml::state<errored>,

      sml::state<parsed> + sml::event<event::probe> [guard::valid_probe{}] /
        action::run_probe = sml::state<probed>,
      sml::state<errored> + sml::event<event::probe> [guard::valid_probe{}] /
        action::run_probe = sml::state<probed>,

      sml::state<uninitialized> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<probed> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<bound> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<parsed> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::probe & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::probe_done{&ev, context_.probed});
      }
    } else if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
      ev.dispatch_error(ev.owner_sm, events::probe_error{&ev, err});
    }
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::bind_storage & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::bind_done{&ev});
      }
    } else if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
      ev.dispatch_error(ev.owner_sm, events::bind_error{&ev, err});
    }
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::parse & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_.last_error;
    if (err == EMEL_OK) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::parse_done{&ev});
      }
    } else if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
      ev.dispatch_error(ev.owner_sm, events::parse_error{&ev, err});
    }
    return accepted && err == EMEL_OK;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  using base_type::raw_sm;

  action::context context_{};
};

}  // namespace emel::parser::gguf
