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
